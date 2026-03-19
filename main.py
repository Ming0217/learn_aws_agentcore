"""
main.py - Customer Support Agent
=================================
This is the core runtime for the AI-powered customer support agent.
It defines the tools the agent can use and wires everything together
using the Strands Agents framework backed by AWS Bedrock.

ARCHITECTURE OVERVIEW:
  User query → Agent (LLM) → selects tool(s) → tool returns data → Agent formats response

STRANDS AGENTS FRAMEWORK:
  Strands is an AWS open-source framework for building LLM agents.
  It handles the tool-calling loop automatically:
    1. Sends user message + tool definitions to the LLM
    2. LLM decides which tool(s) to call and with what arguments
    3. Strands executes the tool and sends results back to the LLM
    4. LLM formulates the final response
  This loop is called the "agentic loop" or "ReAct" (Reason + Act) pattern.

LAB 2 ADDITIONS — AgentCore Memory:
  The agent now has persistent memory across sessions via AWS Bedrock AgentCore Memory.
  This means the agent can recall past customer interactions, preferences, and issues
  even when the process restarts — unlike in-memory conversation history which is lost
  when the session ends.

  Memory flow:
    Customer message → Agent responds → AgentCoreMemorySessionManager auto-saves interaction
    → On next session, relevant memories are retrieved and injected into the LLM context
    → Agent personalizes response based on what it remembers about this customer

SEPARATION OF CONCERNS:
  - main.py          → agent runtime (tools, model, memory config, system prompt)
  - kb_setup.py      → one-time KB setup (S3 download + Bedrock ingestion)
  - create_memories.py → one-time memory setup (create memory store + seed history)
"""

import boto3
import uuid

from boto3.session import Session
from ddgs.exceptions import DDGSException, RatelimitException
from ddgs import DDGS
from strands.tools import tool
from strands.models import BedrockModel
from strands import Agent
from strands_tools import retrieve
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from lab_helpers.lab2_memory import ACTOR_ID
# LEARNING NOTE: ACTOR_ID is the unique identifier for the customer whose memories
# we're retrieving. In production this would come from your auth system (e.g., Cognito sub).
# It's used to namespace memories so each customer's data is isolated from others.

# ---------------------------------------------------------------------------
# AWS Session Setup
# ---------------------------------------------------------------------------
# LEARNING NOTE: Using boto3.Session() (rather than individual client calls)
# is best practice — it respects environment variables, ~/.aws/credentials,
# and IAM role credentials automatically. Always resolve region at runtime.
# ---------------------------------------------------------------------------
boto_session = Session()
region = boto_session.region_name

# ---------------------------------------------------------------------------
# Fetch shared resource IDs from SSM Parameter Store
# ---------------------------------------------------------------------------
# LEARNING NOTE: memory_id was created by create_memories.py and saved to SSM.
# Reading it here at startup decouples the two scripts — create_memories.py
# only needs to run once (or when you want to reset memory), while main.py
# can be restarted freely and will always pick up the correct memory_id.
# This is the standard pattern for sharing resource IDs across scripts/services.
# ---------------------------------------------------------------------------
ssm = boto3.client("ssm")
memory_id = ssm.get_parameter(Name="/app/customersupport/agentcore/memory_id")["Parameter"]["Value"]


# ---------------------------------------------------------------------------
# TOOL 1: get_return_policy
# ---------------------------------------------------------------------------
# LEARNING NOTE: The @tool decorator from Strands converts a regular Python
# function into a tool the LLM can call. Strands reads the function's
# docstring and type hints to generate the tool schema (JSON) that gets
# sent to the LLM, so clear docstrings are important — they guide the LLM
# on WHEN and HOW to use each tool.
# ---------------------------------------------------------------------------
@tool
def get_return_policy(product_category: str) -> str:
    """
    Get return policy information for a specific product category.

    Args:
        product_category: Electronics category (e.g., 'smartphones', 'laptops', 'accessories')

    Returns:
        Formatted return policy details including timeframes and conditions
    """
    # Mock return policy database.
    # LEARNING NOTE: In production, this would call an internal API or query
    # a database (e.g., DynamoDB or RDS). Mocking here keeps the lab self-contained.
    return_policies = {
        "smartphones": {
            "window": "30 days",
            "condition": "Original packaging, no physical damage, factory reset required",
            "process": "Online RMA portal or technical support",
            "refund_time": "5-7 business days after inspection",
            "shipping": "Free return shipping, prepaid label provided",
            "warranty": "1-year manufacturer warranty included",
        },
        "laptops": {
            "window": "30 days",
            "condition": "Original packaging, all accessories, no software modifications",
            "process": "Technical support verification required before return",
            "refund_time": "7-10 business days after inspection",
            "shipping": "Free return shipping with original packaging",
            "warranty": "1-year manufacturer warranty, extended options available",
        },
        "accessories": {
            "window": "30 days",
            "condition": "Unopened packaging preferred, all components included",
            "process": "Online return portal",
            "refund_time": "3-5 business days after receipt",
            "shipping": "Customer pays return shipping under $50",
            "warranty": "90-day manufacturer warranty",
        },
    }

    # Fallback for categories not in the mock database
    default_policy = {
        "window": "30 days",
        "condition": "Original condition with all included components",
        "process": "Contact technical support",
        "refund_time": "5-7 business days after inspection",
        "shipping": "Return shipping policies vary",
        "warranty": "Standard manufacturer warranty applies",
    }

    policy = return_policies.get(product_category.lower(), default_policy)
    return (
        f"Return Policy - {product_category.title()}:\n\n"
        f"• Return window: {policy['window']} from delivery\n"
        f"• Condition: {policy['condition']}\n"
        f"• Process: {policy['process']}\n"
        f"• Refund timeline: {policy['refund_time']}\n"
        f"• Shipping: {policy['shipping']}\n"
        f"• Warranty: {policy['warranty']}"
    )

print("✅ Return policy tool ready")


# ---------------------------------------------------------------------------
# TOOL 2: get_product_info
# ---------------------------------------------------------------------------
# LEARNING NOTE: This is another mock tool. Notice the pattern — each tool
# has a single, focused responsibility. This makes it easier for the LLM
# to pick the right tool and for you to test/debug each one independently.
# ---------------------------------------------------------------------------
@tool
def get_product_info(product_type: str) -> str:
    """
    Get detailed technical specifications and information for electronics products.

    Args:
        product_type: Electronics product type (e.g., 'laptops', 'smartphones', 'headphones', 'monitors')
    Returns:
        Formatted product information including warranty, features, and policies
    """
    # Mock product catalog
    products = {
        "laptops": {
            "warranty": "1-year manufacturer warranty + optional extended coverage",
            "specs": "Intel/AMD processors, 8-32GB RAM, SSD storage, various display sizes",
            "features": "Backlit keyboards, USB-C/Thunderbolt, Wi-Fi 6, Bluetooth 5.0",
            "compatibility": "Windows 11, macOS, Linux support varies by model",
            "support": "Technical support and driver updates included",
        },
        "smartphones": {
            "warranty": "1-year manufacturer warranty",
            "specs": "5G/4G connectivity, 128GB-1TB storage, multiple camera systems",
            "features": "Wireless charging, water resistance, biometric security",
            "compatibility": "iOS/Android, carrier unlocked options available",
            "support": "Software updates and technical support included",
        },
        "headphones": {
            "warranty": "1-year manufacturer warranty",
            "specs": "Wired/wireless options, noise cancellation, 20Hz-20kHz frequency",
            "features": "Active noise cancellation, touch controls, voice assistant",
            "compatibility": "Bluetooth 5.0+, 3.5mm jack, USB-C charging",
            "support": "Firmware updates via companion app",
        },
        "monitors": {
            "warranty": "3-year manufacturer warranty",
            "specs": "4K/1440p/1080p resolutions, IPS/OLED panels, various sizes",
            "features": "HDR support, high refresh rates, adjustable stands",
            "compatibility": "HDMI, DisplayPort, USB-C inputs",
            "support": "Color calibration and technical support",
        },
    }

    product = products.get(product_type.lower())
    if not product:
        return f"Technical specifications for {product_type} not available. Please contact our technical support team for detailed product information and compatibility requirements."

    return (
        f"Technical Information - {product_type.title()}:\n\n"
        f"• Warranty: {product['warranty']}\n"
        f"• Specifications: {product['specs']}\n"
        f"• Key Features: {product['features']}\n"
        f"• Compatibility: {product['compatibility']}\n"
        f"• Support: {product['support']}"
    )

print("✅ get_product_info tool ready")


# ---------------------------------------------------------------------------
# TOOL 3: web_search
# ---------------------------------------------------------------------------
# LEARNING NOTE: This tool gives the agent access to live internet data via
# DuckDuckGo (no API key required). This is useful when the agent needs
# information that isn't in the KB or mock data — e.g., latest firmware
# versions, recent product recalls, or current pricing.
#
# IMPORTANT: In production, be mindful of:
#   - Rate limiting (handled below)
#   - Data quality — web results are unverified, so the LLM should be
#     instructed to treat them as supplementary, not authoritative
#   - Cost/latency — web search adds latency to the agent response
# ---------------------------------------------------------------------------
@tool
def web_search(keywords: str, region: str = "us-en", max_results: int = 5) -> str:
    """Search the web for updated information.

    Args:
        keywords (str): The search query keywords.
        region (str): The search region: wt-wt, us-en, uk-en, ru-ru, etc..
        max_results (int | None): The maximum number of results to return.
    Returns:
        List of dictionaries with search results.
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return results if results else "No results found."
    except RatelimitException:
        # LEARNING NOTE: Always handle rate limits gracefully — return a
        # meaningful message so the agent can inform the user instead of crashing.
        return "Rate limit reached. Please try again later."
    except DDGSException as e:
        return f"Search error: {e}"
    except Exception as e:
        return f"Search error: {str(e)}"

print("✅ Web search tool ready")


# ---------------------------------------------------------------------------
# TOOL 4: get_technical_support
# ---------------------------------------------------------------------------
# LEARNING NOTE: This is the most sophisticated tool — it implements RAG
# (Retrieval-Augmented Generation) by querying the Bedrock Knowledge Base.
#
# RAG PATTERN EXPLAINED:
#   Instead of relying solely on the LLM's training data, RAG retrieves
#   relevant passages from YOUR documents at query time and injects them
#   into the LLM's context. This means:
#     ✅ Answers are grounded in your actual product documentation
#     ✅ No need to fine-tune the model when docs change — just re-sync the KB
#     ✅ Reduces hallucinations for domain-specific questions
#
# HOW THIS TOOL WORKS:
#   1. Looks up the KB ID from SSM Parameter Store (avoids hardcoding)
#   2. Calls the Strands `retrieve` tool to query the Bedrock KB
#   3. Bedrock converts the query to an embedding, finds similar doc chunks,
#      and returns the most relevant passages
#   4. Those passages are returned to the agent, which uses them to answer
# ---------------------------------------------------------------------------
@tool
def get_technical_support(issue_description: str) -> str:
    """
    Retrieve technical support information from the Bedrock Knowledge Base.
    Use this for troubleshooting, setup guides, maintenance tips, and detailed technical assistance.

    Args:
        issue_description: Description of the technical issue or question

    Returns:
        Relevant technical support content from the knowledge base
    """
    try:
        ssm = boto3.client("ssm")
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        region = boto3.Session().region_name

        # Fetch the KB ID from SSM — never hardcode resource IDs
        kb_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/knowledge-base-id")["Parameter"]["Value"]
        print(f"Successfully retrieved KB ID: {kb_id}")

        # Build the tool_use payload for the Strands retrieve tool.
        # LEARNING NOTE: The `retrieve` tool from strands_tools wraps the
        # Bedrock RetrieveAndGenerate API. The `score` threshold (0.4) filters
        # out low-relevance results — tune this based on your use case.
        # Lower score = more results but potentially less relevant.
        tool_use = {
            "toolUseId": "tech_support_query",
            "input": {
                "text": issue_description,
                "knowledgeBaseId": kb_id,
                "region": region,
                "numberOfResults": 3,   # Return top 3 most relevant passages
                "score": 0.4,           # Minimum relevance score threshold
            },
        }

        result = retrieve.retrieve(tool_use)

        if result["status"] == "success":
            return result["content"][0]["text"]
        else:
            return f"Unable to access technical support documentation. Error: {result['content'][0]['text']}"

    except Exception as e:
        print(f"Detailed error in get_technical_support: {str(e)}")
        return f"Unable to access technical support documentation. Error: {str(e)}"

print("✅ Technical support tool ready")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
# LEARNING NOTE: The system prompt is the agent's "personality and rulebook".
# It tells the LLM:
#   - What role it plays (customer support agent)
#   - What tools are available and when to use each one
#   - How to behave (tone, escalation paths)
#
# Good system prompts are specific about tool usage — vague prompts lead to
# the LLM guessing which tool to use, causing incorrect or inconsistent behavior.
# Notice how each tool is explicitly described with its use case.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful and professional customer support assistant for an electronics e-commerce company.
Your role is to:
- Provide accurate information using the tools available to you
- Support the customer with technical information and product specifications, and maintenance questions
- Be friendly, patient, and understanding with customers
- Always offer additional help after answering questions
- If you can't help with something, direct customers to the appropriate contact

You have access to the following tools:
1. get_return_policy() - For warranty and return policy questions
2. get_product_info() - To get information about a specific product
3. web_search() - To access current technical documentation, or for updated information
4. get_technical_support() - For troubleshooting issues, setup guides, maintenance tips, and detailed technical assistance

For any technical problems, setup questions, or maintenance concerns, always use the get_technical_support() tool
as it contains our comprehensive technical documentation and step-by-step guides.

Always use the appropriate tool to get accurate, up-to-date information rather than making assumptions about
electronic products or specifications."""


session_id = uuid.uuid4()
# LEARNING NOTE: A new session_id is generated on every run (uuid4 = random UUID).
# This tells AgentCore Memory that this is a new conversation session.
# Short-term memory (within a session) is scoped to this ID.
# Long-term memory (preferences, semantic facts) persists across session IDs,
# keyed by ACTOR_ID instead — that's what enables cross-session personalization.

# ---------------------------------------------------------------------------
# Memory Configuration
# ---------------------------------------------------------------------------
# LEARNING NOTE: AgentCoreMemoryConfig wires the agent to the Bedrock AgentCore
# Memory store created by create_memories.py. Key concepts:
#
#   memory_id   → identifies WHICH memory store to use (fetched from SSM)
#   session_id  → identifies the current conversation session (new each run)
#   actor_id    → identifies the customer (used to namespace their memories)
#
#   retrieval_config → controls HOW memories are retrieved per namespace:
#     - top_k: max number of memory chunks to retrieve per query
#     - relevance_score: minimum similarity threshold (0.0-1.0)
#       Lower = more results but potentially less relevant
#       0.2 is intentionally permissive here to maximize recall
#
#   Namespaces separate different types of memories:
#     - semantic/    → factual info extracted from conversations (e.g., "has MacBook Pro")
#     - preferences/ → inferred preferences (e.g., "prefers ThinkPad, budget $1200")
# ---------------------------------------------------------------------------
memory_config = AgentCoreMemoryConfig(
        memory_id=memory_id,
        session_id=str(session_id),
        actor_id=ACTOR_ID,
        retrieval_config={
            "support/customer/{actorId}/semantic/": RetrievalConfig(top_k=3, relevance_score=0.2),
            "support/customer/{actorId}/preferences/": RetrievalConfig(top_k=3, relevance_score=0.2)
        }
    )

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
# LEARNING NOTE: BedrockModel wraps the Bedrock Converse API.
# Key parameters:
#   - model_id: The foundation model to use. Here we use Claude Haiku for
#     speed and cost efficiency. For more complex reasoning, you'd use Sonnet.
#   - temperature: Controls randomness. 0.0 = deterministic, 1.0 = creative.
#     For customer support, 0.3 keeps responses consistent and factual.
#   - region_name: Always pass explicitly to avoid region misconfiguration.
# ---------------------------------------------------------------------------
model = BedrockModel(
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature=0.3,    # Low temperature = more consistent, factual responses
    region_name=region,
)


# ---------------------------------------------------------------------------
# Agent Initialization
# ---------------------------------------------------------------------------
# LEARNING NOTE: The Agent class is the orchestrator. It manages:
#   - The agentic loop (send message → get tool call → execute → repeat)
#   - Conversation memory (maintains context across turns in a session)
#   - Tool registration (makes tools available to the LLM via JSON schema)
#
# LAB 2 CHANGE: session_manager is now passed in.
#   AgentCoreMemorySessionManager hooks into the Strands agent lifecycle:
#     - BEFORE each turn: retrieves relevant memories for this customer and
#       injects them into the LLM context automatically
#     - AFTER each turn: saves the new interaction to the memory store
#   This is the "memory hook" pattern — you don't call memory APIs manually,
#   the session manager handles it transparently behind the scenes.
#
# The order of tools in the list doesn't affect behavior — the LLM picks
# the right tool based on the docstrings and system prompt guidance.
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    session_manager=AgentCoreMemorySessionManager(memory_config, region),
    tools=[
        get_product_info,       # Tool 1: Mock product specs lookup
        get_return_policy,      # Tool 2: Mock return policy lookup
        web_search,             # Tool 3: Live web search via DuckDuckGo
        get_technical_support,  # Tool 4: RAG-based KB retrieval via Bedrock
    ],
    system_prompt=SYSTEM_PROMPT,
)

print("✅ Customer Support Agent created successfully!")


# ---------------------------------------------------------------------------
# Test Queries
# ---------------------------------------------------------------------------
# LEARNING NOTE: These two queries specifically test memory recall, not just
# tool usage. They only work correctly if create_memories.py was run first
# to seed the customer's interaction history.
#
# Query 1 — tests preference-based personalization:
#   The agent should recall from seeded history that this customer is into
#   competitive FPS gaming and needs low latency, and tailor the recommendation.
#
# Query 2 — tests semantic memory recall:
#   The agent should recall the customer's stated preference for ThinkPad,
#   16GB RAM minimum, Linux compatibility, and $1200 budget — without the
#   customer repeating any of that in this session.
#
# If memory is working correctly, responses will be personalized.
# If memory isn't working, responses will be generic — useful for debugging.
# ---------------------------------------------------------------------------
print("🎧 Testing headphone recommendation with customer memory...\n\n")
response1 = agent("Which headphones would you recommend?")

print("\n💻 Testing laptop preference recall...\n\n")
response2 = agent("What is my preferred laptop brand and requirements?")