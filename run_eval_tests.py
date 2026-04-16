"""
run_eval_tests.py - Generate Test Interactions for Evaluation
==============================================================
Invokes the deployed agent with various customer support scenarios to
generate traces that the online evaluators will score.

WHAT THIS DOES:
  Runs 5 test scenarios against the production agent, each designed to
  exercise different tools and capabilities:
    1. Product info query → should use get_product_info
    2. Technical support → should use get_technical_support (RAG)
    3. Return policy → should use get_return_policy
    4. Complex multi-tool → should use multiple tools in one session
    5. Capability listing → should list all available tools

WHY THESE SCENARIOS MATTER FOR EVALUATION:
  Each scenario tests a different evaluator dimension:
    - GoalSuccessRate: Did the agent actually answer the question?
    - Correctness: Is the product/policy info accurate?
    - ToolSelectionAccuracy: Did it pick get_product_info for product queries
      and get_technical_support for troubleshooting?

  After running these, check CloudWatch → GenAI Observability → Bedrock AgentCore
  to see the evaluation scores. Results may take a few minutes to appear.

USAGE:
  python run_eval_tests.py
"""

import uuid
from pathlib import Path
from boto3.session import Session
from bedrock_agentcore_starter_toolkit import Runtime

from lab_helpers.utils import get_or_create_cognito_pool, get_ssm_parameter
from lab_helpers.lab2_memory import ACTOR_ID

boto_session = Session()
REGION = boto_session.region_name


def get_runtime():
    """Reconnect to the deployed Runtime."""
    runtime = Runtime()
    runtime.configure(
        entrypoint="lab_helpers/lab4_runtime.py",
        execution_role=get_ssm_parameter("/app/customersupport/agentcore/runtime_iam_role"),
        auto_create_ecr=False,
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


def invoke_agent(runtime_client, bearer_token, prompt, session_id=None):
    """Invoke the agent and return the response."""
    if not session_id:
        session_id = str(uuid.uuid4())

    response = runtime_client.invoke(
        payload={"prompt": prompt, "actor_id": ACTOR_ID},
        session_id=session_id,
        bearer_token=bearer_token,
    )
    return response, session_id


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------
# LEARNING NOTE: Each scenario is designed to exercise a specific tool and
# evaluator dimension. The evaluators will score each session independently.
# Running diverse scenarios gives you a comprehensive view of agent quality.
# ---------------------------------------------------------------------------
TEST_SCENARIOS = [
    {
        "name": "Product Information Query",
        "prompt": "I need information about the Gaming Console Pro. What are its specifications and price?",
        "expected_tool": "get_product_info",
    },
    {
        "name": "Technical Support Request",
        "prompt": "My laptop won't start up. Can you help me troubleshoot this issue?",
        "expected_tool": "get_technical_support",
    },
    {
        "name": "Return Policy Inquiry",
        "prompt": "I bought a smartphone last week but it's not working properly. What's your return policy?",
        "expected_tool": "get_return_policy",
    },
    {
        "name": "Complex Multi-Tool Query",
        "prompt": "I need help with my Gaming Console Pro. First, can you tell me about its warranty? Then I need technical support for connection issues.",
        "expected_tool": "check_warranty_status + get_technical_support",
    },
    {
        "name": "General Capability Query",
        "prompt": "What kind of support can you provide? List all your available tools and capabilities.",
        "expected_tool": "none (informational)",
    },
]


def run_tests():
    print(f"Running {len(TEST_SCENARIOS)} evaluation test scenarios\n")

    runtime_client = get_runtime()

    access_token = get_or_create_cognito_pool(refresh_token=True)
    bearer_token = access_token["bearer_token"]
    print("✅ Cognito token ready\n")

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"{'=' * 60}")
        print(f"Test {i}: {scenario['name']}")
        print(f"Expected tool: {scenario['expected_tool']}")
        print(f"{'=' * 60}")

        try:
            response, session_id = invoke_agent(
                runtime_client, bearer_token, scenario["prompt"]
            )
            print(f"Session ID: {session_id}")
            print(f"Response:\n{response['response']}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

    print("✅ All test scenarios completed!")
    print("\n📊 Check evaluation results in:")
    print("   CloudWatch → GenAI Observability → Bedrock AgentCore")
    print("   Results may take a few minutes to appear.")


if __name__ == "__main__":
    run_tests()
