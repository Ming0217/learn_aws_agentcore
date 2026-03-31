"""
main.py - Customer Support Agent (Lab 3)
==========================================
This is the core runtime for the AI-powered customer support agent.

LAB 3 CHANGES — AgentCore Gateway Integration:
  The agent now uses a MIX of local tools and remote MCP tools:
    - LOCAL tools (run in-process): get_product_info, get_return_policy, get_technical_support
      These are specific to this agent and don't need to be shared.
    - REMOTE tools (via AgentCore Gateway): web_search, check_warranty_status
      These are centralized behind the Gateway so any agent can use them.

  The MCPClient connects to the Gateway over HTTP with a JWT token from Cognito.
  The agent doesn't know or care which tools are local vs remote — Strands
  handles both transparently through the same tool interface.

ARCHITECTURE:
  User query → Agent (LLM) → picks tool →
    if local tool: executes in-process (same as Lab 1)
    if MCP tool:   MCPClient → Gateway → Lambda → result flows back

HOW MCPClient, GATEWAY, AND LAMBDA TALK TO EACH OTHER:

  Think of it like a restaurant:
    - MCPClient = the customer placing an order
    - Gateway   = the front desk that takes the order, checks your reservation (auth),
                  and routes it to the right kitchen station
    - Lambda    = the kitchen that actually cooks the food and sends it back

  They form a chain, each speaking a different protocol:

    MCPClient ←── MCP protocol (HTTP) ──→ Gateway ←── AWS Lambda Invoke ──→ Lambda
    (speaks MCP)                          (translates)                      (does the work)

  Runtime flow when a user asks "check my warranty":

    1. Agent starts → MCPClient connects to Gateway via HTTP
    2. Gateway returns tool schemas (from api_spec.json) → Agent knows what tools exist
    3. User asks a question → LLM decides to call "check_warranty_status"
    4. MCPClient sends the tool call to the Gateway (HTTP POST + JWT token)
    5. Gateway validates JWT, then invokes the Lambda (AWS Lambda Invoke API)
    6. Lambda runs check_warranty_status(), returns results
    7. Results flow back: Lambda → Gateway → MCPClient → Agent → LLM → User

  Where auth fits in:

    Cognito issues JWT → MCPClient includes it in HTTP header → Gateway validates
    before routing. Without a valid token, the Gateway rejects the request.

  Why the Gateway sits in the middle (3 things neither MCPClient nor Lambda can do):
    1. Protocol translation: MCP (what agents speak) → Lambda Invoke (what AWS speaks)
    2. Authentication: validates JWT before forwarding any request
    3. Routing: one Gateway can have multiple targets (Lambdas, REST APIs, etc.)
       and routes tool calls to the right one based on tool name

SETUP ORDER:
  1. python deploy_infrastructure.py  (one-time: CloudFormation stack)
  2. python setup_gateway.py          (one-time: Gateway + Lambda target)
  3. python create_memories.py        (one-time: memory store + seed data)
  4. python main.py                   (run the agent)
"""

import uuid
import boto3
from boto3.session import Session

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

# Import local tools from the shared module (Lab 1 tools, reusable)
from lab_helpers.lab1_strands_agent import (
    get_product_info,
    get_return_policy,
    get_technical_support,
    SYSTEM_PROMPT,
)
from lab_helpers.lab2_memory import ACTOR_ID
from lab_helpers.utils import get_or_create_cognito_pool, get_ssm_parameter


# ---------------------------------------------------------------------------
# AWS Session Setup
# ---------------------------------------------------------------------------
boto_session = Session()
REGION = boto_session.region_name

# ---------------------------------------------------------------------------
# Fetch shared resource IDs from SSM
# ---------------------------------------------------------------------------
# LEARNING NOTE: All resource IDs are stored in SSM by their respective
# setup scripts. main.py never creates resources — it only reads IDs.
# This keeps the runtime script clean and focused on agent logic.
# ---------------------------------------------------------------------------
ssm = boto3.client("ssm")
memory_id = ssm.get_parameter(Name="/app/customersupport/agentcore/memory_id")["Parameter"]["Value"]
gateway_url = ssm.get_parameter(Name="/app/customersupport/agentcore/gateway_url")["Parameter"]["Value"]

print(f"✅ Memory ID: {memory_id}")
print(f"✅ Gateway URL: {gateway_url}")


# ---------------------------------------------------------------------------
# MCP Client Setup
# ---------------------------------------------------------------------------
# LEARNING NOTE: MCPClient wraps the MCP protocol for Strands.
# It connects to the Gateway via streamablehttp_client (HTTP transport).
# The JWT token is passed in the Authorization header on every request.
#
# We create a fresh MCPClient inside create_agent() rather than at module
# level, because:
#   1. JWT tokens expire — a fresh token is needed for each invocation
#   2. The MCP connection is stateful — reusing across `with` blocks can fail
#
# MCPClient's role in the chain:
#   MCPClient ──HTTP+JWT──→ Gateway ──Lambda Invoke──→ Lambda
#   (you are here)
#
# MCPClient does two things:
#   1. Tool discovery: asks Gateway "what tools do you have?" → gets schemas
#   2. Tool execution: sends "call web_search with {keywords: '...'}" → gets result
#
# The agent code never calls Lambda directly. MCPClient abstracts the entire
# remote tool call into the same interface as a local @tool function.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Memory Configuration (same as Lab 2)
# ---------------------------------------------------------------------------
session_id = uuid.uuid4()

memory_config = AgentCoreMemoryConfig(
    memory_id=memory_id,
    session_id=str(session_id),
    actor_id=ACTOR_ID,
    retrieval_config={
        "support/customer/{actorId}/semantic/": RetrievalConfig(top_k=3, relevance_score=0.2),
        "support/customer/{actorId}/preferences/": RetrievalConfig(top_k=3, relevance_score=0.2),
    },
)


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
model = BedrockModel(
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature=0.3,
    region_name=REGION,
)


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------
# LEARNING NOTE: In Lab 3, the agent is created inside a function wrapped
# in `with mcp_client:`. This is because:
#   1. The MCP connection must be open when listing/calling remote tools
#   2. `mcp_client.list_tools_sync()` discovers tools from the Gateway
#   3. Local tools + MCP tools are combined into a single tools list
#   4. The agent doesn't distinguish between local and remote tools
#
# The `with` block ensures the MCP connection is properly opened and closed.
# Each call to create_agent() gets a fresh connection — important because
# JWT tokens can expire and connections can go stale.
# ---------------------------------------------------------------------------
def create_agent(prompt):
    """
    Create an agent with both local and MCP tools, then run a prompt.

    LEARNING NOTE: A fresh MCPClient and bearer token are created on each call.
    This ensures the JWT hasn't expired and the MCP connection is clean.
    Local tools and MCP tools are concatenated into a single list — Strands
    treats them identically. The LLM doesn't know which are local vs remote.
    """
    try:
        # Get a fresh bearer token for each invocation
        fresh_cognito = get_or_create_cognito_pool(refresh_token=True)

        mcp_client = MCPClient(
            lambda: streamablehttp_client(
                gateway_url,
                headers={"Authorization": f"Bearer {fresh_cognito['bearer_token']}"},
            )
        )

        with mcp_client:
            # Combine local tools (from lab1_strands_agent) with remote MCP tools
            tools = [
                get_product_info,       # Local: mock product specs
                get_return_policy,      # Local: mock return policy
                get_technical_support,  # Local: RAG via Bedrock KB
            ] + mcp_client.list_tools_sync()  # Remote: web_search + check_warranty_status

            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
            )
            response = agent(prompt)
            return response
    except Exception as e:
        raise e


print("✅ Customer Support Agent ready!")


# ---------------------------------------------------------------------------
# Test Queries
# ---------------------------------------------------------------------------
# LEARNING NOTE: These test cases exercise different tool types:
#   - "List all tools" → verifies the agent sees all 5 tools (3 local + 2 MCP)
#   - iPhone heating → should use get_technical_support (local, RAG)
#   - Warranty check → should use check_warranty_status (MCP, via Gateway → Lambda)
#   - Warranty guidelines → should use get_technical_support (local, RAG)
#   - ThinkPad blue screen → should use get_technical_support (local, RAG)
#   - CPU installation → should use get_technical_support (local, RAG)
# ---------------------------------------------------------------------------
test_prompts = [
    "List all of your tools",
    "I bought an iphone 14 last month. I don't like it because it heats up. How do I solve it?",
    "I have a Gaming Console Pro device, I want to check my warranty status, warranty serial number is MNO33333333.",
    "What are the warranty support guidelines?",
    "How can I fix Lenovo Thinkpad with a blue screen",
    "Tell me detailed information about the technical documentation on installing a new CPU",
]


def test_agent_responses(prompts):
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest Case {i}: {prompt}")
        print("-" * 50)
        try:
            response = create_agent(prompt)
            print(response)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    test_agent_responses(test_prompts)
    print("\n✅ Testing completed!")
