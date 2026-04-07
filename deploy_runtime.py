"""
deploy_runtime.py - Deploy Agent to AgentCore Runtime
=======================================================
Packages and deploys the agent as a managed container service on AgentCore Runtime.
Run this AFTER setup_gateway.py and create_memories.py have completed.

WHAT THIS DOES:
  1. Creates the IAM execution role for the Runtime container
  2. Configures the deployment (generates Dockerfile + .bedrock_agentcore.yaml)
  3. Launches the agent (builds Docker image, pushes to ECR, deploys to Runtime)
  4. Waits for the deployment to reach READY status
  5. Saves the Runtime ARN to SSM

HOW AGENTCORE RUNTIME DEPLOYMENT WORKS:
  configure() → generates Dockerfile based on your entrypoint + requirements.txt
  launch()    → triggers a CodeBuild pipeline that:
                  1. Builds the Docker image
                  2. Pushes it to ECR (auto-created if auto_create_ecr=True)
                  3. Deploys the container to AgentCore Runtime
                  4. Sets up auto-scaling, health checks, and observability

  The whole process takes ~5-10 minutes. The Runtime then runs your agent
  as a persistent HTTP service at a managed endpoint.

PREREQUISITES:
  - Docker must be installed and running (used by the build process)
  - setup_gateway.py must have been run (Runtime needs the Gateway URL)
  - create_memories.py must have been run (Runtime needs the memory_id)

USAGE:
  python deploy_runtime.py
"""

import time
import boto3
from boto3.session import Session
from bedrock_agentcore_starter_toolkit import Runtime
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore.memory.constants import StrategyType

from lab_helpers.utils import (
    create_agentcore_runtime_execution_role,
    get_or_create_cognito_pool,
    get_ssm_parameter,
    put_ssm_parameter,
)
from lab_helpers.lab2_memory import ACTOR_ID

boto_session = Session()
REGION = boto_session.region_name


# ---------------------------------------------------------------------------
# Step 1: Ensure Memory exists
# ---------------------------------------------------------------------------
# LEARNING NOTE: The Runtime container needs MEMORY_ID as an env var at launch.
# We use get_or_create_memory here as a safety net in case create_memories.py
# wasn't run. In production, you'd always run setup scripts in order.
# ---------------------------------------------------------------------------
def ensure_memory():
    """Get or create the AgentCore Memory store and return its ID."""
    memory_name = "CustomerSupportMemory"
    memory_manager = MemoryManager(region_name=REGION)
    memory = memory_manager.get_or_create_memory(
        name=memory_name,
        strategies=[
            {
                StrategyType.USER_PREFERENCE.value: {
                    "name": "CustomerPreferences",
                    "description": "Captures customer preferences and behavior",
                    "namespaces": ["support/customer/{actorId}/preferences/"],
                }
            },
            {
                StrategyType.SEMANTIC.value: {
                    "name": "CustomerSupportSemantic",
                    "description": "Stores facts from conversations",
                    "namespaces": ["support/customer/{actorId}/semantic/"],
                }
            },
        ],
    )
    memory_id = memory["id"]
    print(f"✅ Memory ID: {memory_id}")
    return memory_id


# ---------------------------------------------------------------------------
# Step 2: Configure the Runtime deployment
# ---------------------------------------------------------------------------
# LEARNING NOTE: configure() does NOT deploy anything yet. It:
#   - Reads your entrypoint file and requirements.txt
#   - Generates a Dockerfile tailored to your code
#   - Generates .bedrock_agentcore.yaml with deployment config
#   - Sets up JWT auth using the same Cognito pool as the Gateway
#
# request_header_configuration allowlists headers that get forwarded to
# your @app.entrypoint function. Authorization MUST be allowlisted for
# JWT token propagation to the Gateway to work.
# ---------------------------------------------------------------------------
def configure_runtime(execution_role_arn):
    """Configure the AgentCore Runtime deployment."""
    agentcore_runtime = Runtime()

    response = agentcore_runtime.configure(
        entrypoint="lab_helpers/lab4_runtime.py",
        execution_role=execution_role_arn,
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=REGION,
        agent_name="customer_support_agent",
        # JWT auth — same Cognito pool used by the Gateway
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
        # Allowlist Authorization header so JWT token reaches the entrypoint function
        request_header_configuration={
            "requestHeaderAllowlist": [
                "Authorization",  # Required for JWT propagation to Gateway
                "X-Amzn-Bedrock-AgentCore-Runtime-Custom-H1",
            ]
        },
    )

    print(f"✅ Runtime configured: {response}")
    return agentcore_runtime


# ---------------------------------------------------------------------------
# Step 3: Launch the agent
# ---------------------------------------------------------------------------
# LEARNING NOTE: launch() triggers the full build + deploy pipeline:
#   1. Builds Docker image using the generated Dockerfile
#   2. Pushes image to ECR (auto-created because auto_create_ecr=True)
#   3. Creates the AgentCore Runtime endpoint
#   4. Starts the container
#
# MEMORY_ID is passed as an env var so the container can access it at startup.
# auto_update_on_conflict=True means re-running this script updates the
# existing Runtime instead of failing with a conflict error.
# ---------------------------------------------------------------------------
def launch_runtime(agentcore_runtime, memory_id):
    """Build and deploy the agent container to AgentCore Runtime."""
    print("🚀 Launching agent to AgentCore Runtime (this takes 5-10 minutes)...")

    launch_result = agentcore_runtime.launch(
        env_vars={"MEMORY_ID": memory_id},
        auto_update_on_conflict=True,
    )

    agent_arn = launch_result.agent_arn
    print(f"✅ Launch initiated. Agent ARN: {agent_arn}")

    # Save ARN to SSM for use by invoke_runtime.py and teardown.py
    put_ssm_parameter("/app/customersupport/agentcore/runtime_arn", agent_arn)
    return agentcore_runtime, agent_arn


# ---------------------------------------------------------------------------
# Step 4: Wait for READY status
# ---------------------------------------------------------------------------
# LEARNING NOTE: Deployment is asynchronous — launch() returns immediately
# while the build pipeline runs in the background. We poll status() until
# the Runtime reaches a terminal state (READY or FAILED).
# This is the same async polling pattern used in kb_setup.py for KB ingestion.
# ---------------------------------------------------------------------------
def wait_for_ready(agentcore_runtime):
    """Poll deployment status until READY or FAILED."""
    terminal_states = ["READY", "CREATE_FAILED", "DELETE_FAILED", "UPDATE_FAILED"]

    status_response = agentcore_runtime.status()
    status = status_response.endpoint["status"]

    while status not in terminal_states:
        print(f"⏳ Deployment status: {status}...")
        time.sleep(15)
        status_response = agentcore_runtime.status()
        status = status_response.endpoint["status"]

    if status == "READY":
        print(f"✅ Agent is READY!")
    else:
        print(f"❌ Deployment failed with status: {status}")

    return status


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Deploying agent to AgentCore Runtime in {REGION}\n")

    print("=== Step 1: Ensuring Memory exists ===")
    memory_id = ensure_memory()

    print("\n=== Step 2: Creating IAM execution role ===")
    execution_role_arn = create_agentcore_runtime_execution_role()
    print(f"✅ Execution role: {execution_role_arn}")

    print("\n=== Step 3: Configuring Runtime deployment ===")
    agentcore_runtime = configure_runtime(execution_role_arn)

    print("\n=== Step 4: Launching agent ===")
    agentcore_runtime, agent_arn = launch_runtime(agentcore_runtime, memory_id)

    print("\n=== Step 5: Waiting for READY status ===")
    status = wait_for_ready(agentcore_runtime)

    if status == "READY":
        print("\n🎉 Deployment complete!")
        print(f"Agent ARN: {agent_arn}")
        print("\nYou can now run: python invoke_runtime.py")
