"""
invoke_runtime.py - Test the Deployed AgentCore Runtime Agent
==============================================================
Invokes the production agent running on AgentCore Runtime and demonstrates
session continuity and memory persistence across calls.

WHAT THIS SHOWS:
  1. Basic invocation — agent uses all 5 tools (3 local + 2 MCP)
  2. Session continuity — same session_id maintains conversation context
  3. New user isolation — different actor_id gets no memory from other users

HOW INVOCATION WORKS:
  agentcore_runtime.invoke() sends an HTTP POST to the Runtime endpoint:
    - payload: {"prompt": "...", "actor_id": "..."}
    - bearer_token: JWT from Cognito (authenticates the caller)
    - session_id: UUID that scopes the conversation

  The Runtime validates the JWT, routes to your @app.entrypoint function,
  and returns the response. All of this happens over HTTPS — no AWS SDK
  calls needed from the client side.

USAGE:
  python invoke_runtime.py
"""

import uuid
import boto3
from boto3.session import Session
from bedrock_agentcore_starter_toolkit import Runtime

from lab_helpers.utils import get_or_create_cognito_pool, get_ssm_parameter
from lab_helpers.lab2_memory import ACTOR_ID

boto_session = Session()
REGION = boto_session.region_name


def get_runtime():
    """Reconnect to the existing deployed Runtime.

    LEARNING NOTE: Runtime() doesn't automatically load .bedrock_agentcore.yaml.
    We need to call configure() again to restore the in-memory state so invoke()
    knows which Runtime endpoint to call. configure() is idempotent — it won't
    redeploy anything, it just reads the existing config and sets up the object.
    """
    from lab_helpers.utils import get_ssm_parameter, create_agentcore_runtime_execution_role

    runtime = Runtime()
    runtime.configure(
        entrypoint="lab_helpers/lab4_runtime.py",
        execution_role=get_ssm_parameter("/app/customersupport/agentcore/runtime_iam_role"),
        auto_create_ecr=False,  # Already exists — don't recreate
        requirements_file="requirements.txt",
        region=REGION,
        agent_name="customer_support_agent",
        authorizer_configuration={
            "customJWTAuthorizer": {
                "allowedClients": [
                    get_ssm_parameter("/app/customersupport/agentcore/client_id")
                ],
                "discoveryUrl": get_ssm_parameter(
                    "/app/customersupport/agentcore/cognito_discovery_url"
                ),
            }
        },
        request_header_configuration={
            "requestHeaderAllowlist": ["Authorization"]
        },
    )
    return runtime


def run_tests():
    agentcore_runtime = get_runtime()

    # Get a fresh JWT token — required for every invocation
    access_token = get_or_create_cognito_pool(refresh_token=True)
    bearer_token = access_token["bearer_token"]
    print("✅ Cognito token ready\n")

    # ---------------------------------------------------------------------------
    # Test 1: Basic invocation — verify all tools are available
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("Test 1: List all tools")
    print("=" * 60)
    session_id = uuid.uuid4()

    response = agentcore_runtime.invoke(
        {"prompt": "List all of your tools", "actor_id": ACTOR_ID},
        bearer_token=bearer_token,
        session_id=str(session_id),
    )
    print(response["response"])

    # ---------------------------------------------------------------------------
    # Test 2: Session continuity — same session_id, follow-up question
    # ---------------------------------------------------------------------------
    # LEARNING NOTE: Passing the same session_id means the Runtime maintains
    # conversation context between calls. The agent "remembers" what was said
    # earlier in this session. This is different from AgentCore Memory (which
    # persists across sessions) — session context is short-term, in-session only.
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Session continuity (same session_id)")
    print("=" * 60)

    response = agentcore_runtime.invoke(
        {
            "prompt": "Tell me detailed information about the technical documentation on installing a new CPU",
            "actor_id": ACTOR_ID,
        },
        bearer_token=bearer_token,
        session_id=str(session_id),  # Same session as Test 1
    )
    print(response["response"])

    # ---------------------------------------------------------------------------
    # Test 3: New session — warranty check via MCP tool
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: Warranty check (new session, MCP tool)")
    print("=" * 60)
    session_id2 = uuid.uuid4()  # New session ID = fresh conversation context

    response = agentcore_runtime.invoke(
        {
            "prompt": "I have a Gaming Console Pro device, I want to check my warranty status, warranty serial number is MNO33333333.",
            "actor_id": ACTOR_ID,
        },
        bearer_token=bearer_token,
        session_id=str(session_id2),
    )
    print(response["response"])

    # ---------------------------------------------------------------------------
    # Test 4: Memory recall — agent should remember customer preferences
    # ---------------------------------------------------------------------------
    # LEARNING NOTE: This tests AgentCore Memory (long-term, cross-session).
    # The agent was seeded with this customer's history in create_memories.py.
    # Even though this is a new session, the agent should recall preferences
    # because they're stored in the Memory store keyed by actor_id.
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 4: Memory recall (cross-session personalization)")
    print("=" * 60)
    session_id3 = uuid.uuid4()

    response = agentcore_runtime.invoke(
        {"prompt": "Which headphones would you recommend?", "actor_id": ACTOR_ID},
        bearer_token=bearer_token,
        session_id=str(session_id3),
    )
    print(response["response"])

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    run_tests()
