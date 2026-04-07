"""
lab_helpers/lab4_runtime.py - AgentCore Runtime Entrypoint
============================================================
This file IS the production agent. It wraps the same agent logic from
Labs 1-3 inside a BedrockAgentCoreApp, which turns it into a managed
HTTP service that AgentCore Runtime can host.

THE 4 KEY LINES (marked with #### AGENTCORE RUNTIME ####):
  1. Import BedrockAgentCoreApp
  2. Initialize the app: app = BedrockAgentCoreApp()
  3. Decorate the handler: @app.entrypoint
  4. Start the server: app.run()

WHAT BedrockAgentCoreApp DOES BEHIND THE SCENES:
  - Creates an HTTP server on port 8080
  - Implements /invocations endpoint (handles agent requests)
  - Implements /ping endpoint (health checks for the container)
  - Handles content types, error formatting, and AWS standards
  - Instruments the code with OpenTelemetry for observability

HOW THE JWT TOKEN FLOWS IN PRODUCTION:
  Client → sends JWT in Authorization header
  → AgentCore Runtime validates JWT (same Cognito pool as Gateway)
  → Runtime passes headers to your @app.entrypoint function via context
  → Your code extracts the token and forwards it to the Gateway
  → Gateway validates the same token again for tool access

  This is called "token propagation" — the same JWT that authenticates
  the user to the Runtime also authenticates them to the Gateway tools.
  No separate auth step needed.

DIFFERENCE FROM main.py:
  main.py:          runs locally, you call it directly, one session
  lab4_runtime.py:  runs in a container, AgentCore calls it via HTTP,
                    handles multiple concurrent sessions, always on

MEMORY_ID is passed as an environment variable (not SSM) because the
container needs it at startup before any SSM calls are made.
"""

import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp  #### AGENTCORE RUNTIME - LINE 1 ####
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
import boto3
from strands.models import BedrockModel
from lab_helpers.utils import get_ssm_parameter
from lab_helpers.lab1_strands_agent import (
    get_return_policy,
    get_product_info,
    get_technical_support,
    SYSTEM_PROMPT,
    MODEL_ID,
)
from lab_helpers.lab2_memory import ACTOR_ID, SESSION_ID
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

REGION = boto3.session.Session().region_name

# Model is initialized once at container startup (not per request) for efficiency.
# LEARNING NOTE: Lambda-style cold start optimization — expensive initializations
# happen outside the handler so they're reused across invocations.
model = BedrockModel(model_id=MODEL_ID)

# MEMORY_ID comes from an environment variable set during agentcore_runtime.launch().
# LEARNING NOTE: Environment variables are the standard way to pass config to containers.
# We can't use SSM here at module load time because the container may not have
# network access during initialization.
memory_id = os.environ.get("MEMORY_ID")
if not memory_id:
    raise Exception("Environment variable MEMORY_ID is required")

# Initialize the AgentCore Runtime App
app = BedrockAgentCoreApp()  #### AGENTCORE RUNTIME - LINE 2 ####


@app.entrypoint  #### AGENTCORE RUNTIME - LINE 3 ####
async def invoke(payload, context=None):
    """
    AgentCore Runtime entrypoint — called on every agent invocation.

    Args:
        payload: dict from the HTTP request body. We expect:
                   - "prompt": the user's message
                   - "actor_id": customer identifier (optional, defaults to ACTOR_ID)
        context: AgentCore Runtime context object. Provides:
                   - context.session_id: unique ID for this conversation session
                   - context.request_headers: HTTP headers from the caller
                     (includes Authorization with the JWT token)

    LEARNING NOTE: The function is async because AgentCore Runtime uses
    asyncio internally. The @app.entrypoint decorator handles the event loop.

    TOKEN PROPAGATION:
      The JWT token arrives in context.request_headers["Authorization"].
      We pass it directly to the MCPClient headers so the Gateway can
      validate it. The same token authenticates both the Runtime call
      AND the Gateway tool calls — no re-authentication needed.
    """
    user_input = payload.get("prompt", "")
    # session_id from context ensures each conversation is isolated.
    # Different users get different session_ids even if they call simultaneously.
    session_id = context.session_id
    actor_id = payload.get("actor_id", ACTOR_ID)

    # Extract JWT token from request headers for Gateway auth propagation
    request_headers = context.request_headers or {}
    auth_header = request_headers.get("Authorization", "")
    print(f"Authorization header present: {bool(auth_header)}")

    # Fetch Gateway URL from SSM (not hardcoded — works across environments)
    existing_gateway_id = get_ssm_parameter("/app/customersupport/agentcore/gateway_id")
    gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)
    gateway_response = gateway_client.get_gateway(gatewayIdentifier=existing_gateway_id)
    gateway_url = gateway_response["gatewayUrl"]

    if not (gateway_url and auth_header):
        return "Error: Missing gateway URL or authorization header"

    try:
        # Pass the propagated JWT token to the Gateway — same token, no re-auth
        mcp_client = MCPClient(lambda: streamablehttp_client(
            url=gateway_url,
            headers={"Authorization": auth_header}
        ))

        with mcp_client:
            tools = (
                [get_product_info, get_return_policy, get_technical_support]
                + mcp_client.list_tools_sync()
            )

            memory_config = AgentCoreMemoryConfig(
                memory_id=memory_id,
                session_id=str(session_id),  # Runtime-provided session ID
                actor_id=actor_id,
                retrieval_config={
                    "support/customer/{actorId}/semantic/": RetrievalConfig(top_k=3, relevance_score=0.2),
                    "support/customer/{actorId}/preferences/": RetrievalConfig(top_k=3, relevance_score=0.2),
                },
            )

            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
            )

            response = agent(user_input)
            # Extract plain text from the response object
            return response.message["content"][0]["text"]

    except Exception as e:
        print(f"Agent error: {str(e)}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run()  #### AGENTCORE RUNTIME - LINE 4 ####
