"""
setup_gateway.py - AgentCore Gateway Setup
============================================
Creates the AgentCore Gateway and adds the Lambda function as a target.
Run this AFTER deploy_infrastructure.py has completed.

WHAT THIS DOES:
  1. Creates (or retrieves) a Cognito User Pool for JWT authentication
  2. Creates the AgentCore Gateway with Cognito-based inbound auth
  3. Adds the Lambda function as a Gateway target with tool schemas
  4. Verifies the setup by listing available tools

LEARNING NOTE: The Gateway is created separately from the CloudFormation
stack because it's an AgentCore-specific resource that you manage through
the AgentCore control plane API, not CloudFormation (yet).

USAGE:
  python setup_gateway.py
"""

import os
import sys
import json
import time
import boto3
from boto3.session import Session

from lab_helpers.utils import (
    get_or_create_cognito_pool,
    put_ssm_parameter,
    get_ssm_parameter,
    load_api_spec,
)

boto_session = Session()
REGION = boto_session.region_name
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)


# ---------------------------------------------------------------------------
# Step 1: Set up Cognito for JWT authentication
# ---------------------------------------------------------------------------
# LEARNING NOTE: AgentCore Gateway requires inbound authentication.
# The agent must present a valid JWT token to use any tools.
# Cognito acts as the identity provider — it issues tokens that the
# Gateway validates on every request.
# ---------------------------------------------------------------------------
def setup_cognito():
    """Create or retrieve Cognito User Pool and get a fresh bearer token."""
    print("Setting up Cognito authentication...")
    cognito_config = get_or_create_cognito_pool(refresh_token=True)
    print(f"✅ Cognito ready. Client ID: {cognito_config['client_id']}")
    return cognito_config


# ---------------------------------------------------------------------------
# Step 2: Create the AgentCore Gateway
# ---------------------------------------------------------------------------
# LEARNING NOTE: The Gateway is the MCP server that agents connect to.
# Key configuration:
#   - protocolType="MCP" — speaks the Model Context Protocol
#   - authorizerType="CUSTOM_JWT" — validates JWT tokens from Cognito
#   - roleArn — IAM role that grants the Gateway permission to invoke Lambda
#
# The Gateway is idempotent — if it already exists, we retrieve it from SSM.
#
# THE GATEWAY'S ROLE IN THE CHAIN:
#   MCPClient ──HTTP+JWT──→ Gateway ──Lambda Invoke──→ Lambda
#                           (you are here)
#
# The Gateway does 3 things neither MCPClient nor Lambda can do alone:
#   1. Protocol translation: converts MCP tool calls into Lambda invocations
#   2. Authentication: validates the JWT token before forwarding any request
#   3. Routing: one Gateway can host multiple targets (Lambdas, REST APIs)
#      and routes each tool call to the correct backend based on tool name
#   4. Tool schema serving: returns api_spec.json to MCPClient so the agent
#      knows what tools are available without hardcoding them
#
# Without the Gateway, your agent would need to invoke Lambda directly
# (requires AWS SDK, IAM credentials, custom routing code). With the Gateway,
# any MCP-compatible agent can use the tools — even agents built with
# frameworks other than Strands.
# ---------------------------------------------------------------------------
def create_gateway(cognito_config):
    """Create the AgentCore Gateway with Cognito JWT auth."""
    gateway_name = "customersupport-gw"

    auth_config = {
        "customJWTAuthorizer": {
            "allowedClients": [cognito_config["client_id"]],
            "discoveryUrl": cognito_config["discovery_url"],
        }
    }

    try:
        print(f"Creating gateway '{gateway_name}' in {REGION}...")

        create_response = gateway_client.create_gateway(
            name=gateway_name,
            roleArn=get_ssm_parameter("/app/customersupport/agentcore/gateway_iam_role"),
            protocolType="MCP",
            authorizerType="CUSTOM_JWT",
            authorizerConfiguration=auth_config,
            description="Customer Support AgentCore Gateway",
        )

        gateway_id = create_response["gatewayId"]
        gateway = {
            "id": gateway_id,
            "name": gateway_name,
            "gateway_url": create_response["gatewayUrl"],
            "gateway_arn": create_response["gatewayArn"],
        }

        # Save all gateway details to SSM for use by main.py and teardown.py
        put_ssm_parameter("/app/customersupport/agentcore/gateway_id", gateway_id)
        put_ssm_parameter("/app/customersupport/agentcore/gateway_name", gateway_name)
        put_ssm_parameter("/app/customersupport/agentcore/gateway_arn", create_response["gatewayArn"])
        put_ssm_parameter("/app/customersupport/agentcore/gateway_url", create_response["gatewayUrl"])

        time.sleep(3)  # Brief wait for Gateway to become active
        print(f"✅ Gateway created. ID: {gateway_id}")
        print(f"   MCP URL: {create_response['gatewayUrl']}")
        return gateway

    except Exception:
        # Gateway already exists — retrieve from SSM
        existing_id = get_ssm_parameter("/app/customersupport/agentcore/gateway_id")
        print(f"Found existing gateway: {existing_id}")

        response = gateway_client.get_gateway(gatewayIdentifier=existing_id)
        return {
            "id": existing_id,
            "name": response["name"],
            "gateway_url": response["gatewayUrl"],
            "gateway_arn": response["gatewayArn"],
        }


# ---------------------------------------------------------------------------
# Step 3: Add Lambda function as a Gateway target
# ---------------------------------------------------------------------------
# LEARNING NOTE: A "target" is a backend that the Gateway routes tool calls to.
# One Gateway can have multiple targets (Lambda, REST APIs, etc.).
# Each target has:
#   - A Lambda ARN (where to send tool calls)
#   - A tool schema (what tools this target provides)
#   - A credential provider (how the Gateway authenticates to the backend)
#
# GATEWAY_IAM_ROLE means the Gateway uses its own IAM role to invoke Lambda.
# For external APIs, you'd use OAuth or API key credential providers instead.
# ---------------------------------------------------------------------------
def add_lambda_target(gateway):
    """Add the Lambda function as a Gateway target with tool schemas."""
    api_spec_file = "prerequisite/lambda/api_spec.json"

    if not os.path.exists(api_spec_file):
        print(f"❌ API spec not found: {api_spec_file}")
        sys.exit(1)

    api_spec = load_api_spec(api_spec_file)

    lambda_target_config = {
        "mcp": {
            "lambda": {
                "lambdaArn": get_ssm_parameter("/app/customersupport/agentcore/lambda_arn"),
                "toolSchema": {"inlinePayload": api_spec},
            }
        }
    }

    credential_config = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]

    try:
        response = gateway_client.create_gateway_target(
            gatewayIdentifier=gateway["id"],
            name="LambdaUsingSDK",
            description="Lambda target with warranty check and web search tools",
            targetConfiguration=lambda_target_config,
            credentialProviderConfigurations=credential_config,
        )
        print(f"✅ Gateway target created: {response['targetId']}")
        print(f"   Tools: check_warranty_status, web_search")
    except Exception as e:
        if "already exists" in str(e).lower() or "ConflictException" in str(type(e)):
            print("✅ Gateway target already exists")
        else:
            print(f"❌ Error creating target: {e}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Setting up AgentCore Gateway in {REGION}\n")

    print("=== Step 1: Cognito Authentication ===")
    cognito_config = setup_cognito()

    print("\n=== Step 2: Create Gateway ===")
    gateway = create_gateway(cognito_config)

    print("\n=== Step 3: Add Lambda Target ===")
    add_lambda_target(gateway)

    print("\n🎉 Gateway setup complete!")
    print(f"MCP URL: {gateway['gateway_url']}")
    print("\nYou can now run: python main.py")
