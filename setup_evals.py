"""
setup_evals.py - Configure Online Evaluation for the Customer Support Agent
=============================================================================
Sets up AgentCore Evaluations to continuously monitor the production agent's
quality in real-time as customers interact with it.

WHAT THIS DOES:
  1. Retrieves the agent ARN from SSM (deployed in Lab 4)
  2. Creates an online evaluation config with 3 built-in evaluators
  3. Verifies the config is active

WHAT IS ONLINE EVALUATION?
  Unlike on-demand evaluation (run manually on selected interactions), online
  evaluation runs AUTOMATICALLY on live production traffic. It consists of:
    - Session sampling: configurable % of sessions to evaluate (100% for demo)
    - Evaluators: LLM-based judges that score each sampled session
    - Dashboard: results flow to CloudWatch GenAI Observability

  The evaluators run asynchronously — they don't slow down the agent's response.
  They analyze the completed session traces after the fact.

BUILT-IN EVALUATORS USED:
  - GoalSuccessRate: Did the agent achieve what the customer asked for?
    High = effective problem-solving. Low = unmet needs or misunderstood requests.

  - Correctness: Is the information factually accurate?
    High = reliable answers. Low = incorrect facts or outdated info.

  - ToolSelectionAccuracy: Did the agent pick the right tool for the job?
    High = proper tool usage. Low = wrong tools or unnecessary calls.

USAGE:
  python setup_evals.py
"""

import json
import boto3
from pathlib import Path
from boto3.session import Session
from bedrock_agentcore_starter_toolkit import Evaluation
from lab_helpers.utils import get_ssm_parameter

boto_session = Session()
REGION = boto_session.region_name


# ---------------------------------------------------------------------------
# Step 1: Retrieve agent info from Lab 4
# ---------------------------------------------------------------------------
def get_agent_info():
    """Get the agent ARN and ID from SSM (saved during Lab 4 deployment)."""
    try:
        agent_arn = get_ssm_parameter("/app/customersupport/agentcore/runtime_arn")
        # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/runtime-id
        agent_id = agent_arn.split(":")[-1].split("/")[-1]
        print(f"✅ Agent ID: {agent_id}")
        print(f"   Agent ARN: {agent_arn}")
        return agent_id, agent_arn
    except Exception as e:
        raise Exception(
            f"Missing agent info from Lab 4. Run deploy_runtime.py first. Error: {e}"
        )


# ---------------------------------------------------------------------------
# Step 2: Create online evaluation configuration
# ---------------------------------------------------------------------------
# LEARNING NOTE: create_online_config() registers an evaluation config with
# AgentCore. Once created, it automatically evaluates sampled sessions.
#
# sampling_rate=100 means every session is evaluated — good for demos but
# expensive in production. For real workloads, 10-20% is typical.
#
# auto_create_execution_role=True lets the toolkit create the IAM role
# that the evaluator needs to read traces and invoke Bedrock models
# (the evaluators themselves are LLM-based — they use a model to judge
# the agent's responses).
# ---------------------------------------------------------------------------
def create_eval_config(agent_id):
    """Create the online evaluation configuration with built-in evaluators."""
    eval_client = Evaluation(region=REGION)

    response = eval_client.create_online_config(
        agent_id=agent_id,
        config_name="customer_support_agent_eval",
        sampling_rate=100,  # 100% for demo; use 10-20% in production
        evaluator_list=[
            "Builtin.GoalSuccessRate",       # Did the agent solve the customer's problem?
            "Builtin.Correctness",           # Is the information factually accurate?
            "Builtin.ToolSelectionAccuracy",  # Did the agent pick the right tool?
        ],
        config_description="Customer support agent online evaluation",
        auto_create_execution_role=True,
    )

    config_id = response["onlineEvaluationConfigId"]
    print(f"\n✅ Online evaluation config created!")
    print(f"   Config ID: {config_id}")
    return eval_client, config_id


# ---------------------------------------------------------------------------
# Step 3: Verify the configuration
# ---------------------------------------------------------------------------
def verify_config(eval_client, config_id):
    """Retrieve and display the evaluation config details."""
    config_details = eval_client.get_online_config(config_id=config_id)
    print("\n📋 Configuration Details:")
    print(json.dumps(config_details, indent=2, default=str))
    return config_details


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Setting up online evaluation in {REGION}\n")

    print("=== Step 1: Retrieving agent info ===")
    agent_id, agent_arn = get_agent_info()

    print("\n=== Step 2: Creating evaluation config ===")
    eval_client, config_id = create_eval_config(agent_id)

    print("\n=== Step 3: Verifying config ===")
    verify_config(eval_client, config_id)

    print("\n🎉 Online evaluation is now active!")
    print("Run 'python run_eval_tests.py' to generate test interactions.")
    print("Then check CloudWatch → GenAI Observability → Bedrock AgentCore for results.")
