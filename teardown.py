"""
teardown.py - Complete Workshop Cleanup
=========================================
Deletes ALL resources created during the workshop, in the correct order.

WHAT THIS DELETES:
  1. AgentCore Gateway targets and Gateway itself
  2. AgentCore Memory store
  3. Cognito User Pool and secrets
  4. SSM parameters created by scripts
  5. CloudFormation stack (DynamoDB, Lambda, IAM roles, S3, KB, etc.)
  6. Staging S3 bucket used for Lambda deployment

LEARNING NOTE: Order matters for cleanup. You must delete child resources
before parents (e.g., Gateway targets before the Gateway, objects in S3
before the bucket). CloudFormation handles ordering for its own resources,
but we need to handle the resources created outside the stack manually.

USAGE:
  python teardown.py
"""

import boto3
import time
from boto3.session import Session

boto_session = Session()
REGION = boto_session.region_name
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

STACK_NAME = "agentcore-workshop"
STAGING_BUCKET = f"{ACCOUNT_ID}-{REGION}-agentcore-staging"

ssm = boto3.client("ssm")
s3 = boto3.client("s3")
cfn = boto3.client("cloudformation")


def get_ssm_safe(name):
    """Get an SSM parameter, returning None if it doesn't exist."""
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1: Delete AgentCore Gateway
# ---------------------------------------------------------------------------
def cleanup_gateway():
    """Delete Gateway targets first, then the Gateway itself.

    LEARNING NOTE: You MUST delete targets before the Gateway.
    If you try to delete the Gateway with active targets, the API will reject it.
    This is a common AWS pattern — child resources must be removed before parents.
    """
    print("🗑️  Cleaning up AgentCore Gateway...")
    gateway_id = get_ssm_safe("/app/customersupport/agentcore/gateway_id")

    if not gateway_id:
        print("   No gateway found, skipping.")
        return

    gateway_client = boto3.client("bedrock-agentcore-control", region_name=REGION)

    try:
        # Delete all targets first
        targets = gateway_client.list_gateway_targets(
            gatewayIdentifier=gateway_id, maxResults=100
        )
        for target in targets.get("items", []):
            print(f"   Deleting target: {target['targetId']}")
            gateway_client.delete_gateway_target(
                gatewayIdentifier=gateway_id, targetId=target["targetId"]
            )

        if targets.get("items"):
            print("   Waiting for target deletions to propagate...")
            time.sleep(5)

        # Delete the gateway
        print(f"   Deleting gateway: {gateway_id}")
        gateway_client.delete_gateway(gatewayIdentifier=gateway_id)
        print("   ✅ Gateway deleted")
    except Exception as e:
        print(f"   ⚠️  Gateway cleanup error: {e}")


# ---------------------------------------------------------------------------
# Step 2: Delete AgentCore Memory
# ---------------------------------------------------------------------------
def cleanup_memory():
    """Delete the AgentCore Memory store.

    LEARNING NOTE: Deleting the memory store removes ALL memories for ALL
    customers — both short-term events and long-term extracted memories.
    In production, you'd never do this. Here it's safe because it's a lab.
    """
    print("🗑️  Cleaning up AgentCore Memory...")
    memory_id = get_ssm_safe("/app/customersupport/agentcore/memory_id")

    if not memory_id:
        print("   No memory store found, skipping.")
        return

    try:
        control_client = boto3.client("bedrock-agentcore-control", region_name=REGION)
        control_client.delete_memory(memoryId=memory_id)
        print(f"   ✅ Memory store deleted: {memory_id}")
    except Exception as e:
        print(f"   ⚠️  Memory cleanup error: {e}")


# ---------------------------------------------------------------------------
# Step 3: Delete Cognito resources
# ---------------------------------------------------------------------------
def cleanup_cognito():
    """Delete Cognito User Pool and associated secrets.

    LEARNING NOTE: Cognito config is stored in Secrets Manager (not SSM)
    because it contains the client_secret. We read the secret to get the
    pool_id, then delete the pool, then delete the secret itself.
    """
    print("🗑️  Cleaning up Cognito...")

    from lab_helpers.utils import (
        get_customer_support_secret,
        cleanup_cognito_resources,
        delete_customer_support_secret,
    )

    try:
        import json
        secret_str = get_customer_support_secret()
        if secret_str:
            config = json.loads(secret_str)
            pool_id = config.get("pool_id")
            if pool_id:
                cleanup_cognito_resources(pool_id)
            delete_customer_support_secret()
            print("   ✅ Cognito resources deleted")
        else:
            print("   No Cognito config found, skipping.")
    except Exception as e:
        print(f"   ⚠️  Cognito cleanup error: {e}")


# ---------------------------------------------------------------------------
# Step 4: Delete SSM parameters created by scripts
# ---------------------------------------------------------------------------
def cleanup_ssm_parameters():
    """Delete SSM parameters that were created outside CloudFormation.

    LEARNING NOTE: CloudFormation deletes the SSM parameters IT created
    (like /app/customersupport/dynamodb/warranty_table_name). But parameters
    created by our Python scripts (setup_gateway.py, create_memories.py, etc.)
    are NOT tracked by CloudFormation, so we must delete them manually.
    Missing a parameter here won't cause errors — just orphaned config data.
    """
    print("🗑️  Cleaning up SSM parameters...")

    # Parameters created by our scripts (not by CloudFormation)
    params_to_delete = [
        "/app/customersupport/agentcore/memory_id",
        "/app/customersupport/agentcore/gateway_id",
        "/app/customersupport/agentcore/gateway_name",
        "/app/customersupport/agentcore/gateway_arn",
        "/app/customersupport/agentcore/gateway_url",
        "/app/customersupport/agentcore/client_id",
        "/app/customersupport/agentcore/pool_id",
        "/app/customersupport/agentcore/cognito_discovery_url",
        "/app/customersupport/agentcore/client_secret",
        "/app/customersupport/agentcore/runtime_execution_role_arn",
        "/app/customersupport/agentcore/policy_engine_id",
    ]

    for param in params_to_delete:
        try:
            ssm.delete_parameter(Name=param)
            print(f"   Deleted: {param}")
        except ssm.exceptions.ParameterNotFound:
            pass
        except Exception as e:
            print(f"   ⚠️  Error deleting {param}: {e}")

    print("   ✅ SSM parameters cleaned up")


# ---------------------------------------------------------------------------
# Step 5: Delete CloudFormation stack
# ---------------------------------------------------------------------------
def cleanup_cloudformation():
    """
    Delete the CloudFormation stack and all its resources.

    LEARNING NOTE: CloudFormation deletes resources in dependency order
    automatically. However, S3 buckets with objects can't be deleted by
    CloudFormation — we empty them first.
    """
    print(f"🗑️  Deleting CloudFormation stack: {STACK_NAME}...")

    try:
        cfn.describe_stacks(StackName=STACK_NAME)
    except Exception:
        print(f"   Stack {STACK_NAME} doesn't exist, skipping.")
        return

    # Empty S3 buckets created by the stack (CloudFormation can't delete non-empty buckets)
    kb_bucket = f"{ACCOUNT_ID}-{REGION}-kb-data-bucket"
    for bucket_name in [kb_bucket]:
        try:
            objects = s3.list_objects_v2(Bucket=bucket_name)
            for obj in objects.get("Contents", []):
                s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
            print(f"   Emptied bucket: {bucket_name}")
        except Exception:
            pass

    try:
        cfn.delete_stack(StackName=STACK_NAME)
        print(f"   Stack deletion initiated. Waiting (this may take a few minutes)...")

        waiter = cfn.get_waiter("stack_delete_complete")
        waiter.wait(
            StackName=STACK_NAME,
            WaiterConfig={"Delay": 15, "MaxAttempts": 60},
        )
        print(f"   ✅ Stack {STACK_NAME} deleted")
    except Exception as e:
        print(f"   ⚠️  Stack deletion error: {e}")
        print("   Check the CloudFormation console for details.")


# ---------------------------------------------------------------------------
# Step 6: Delete staging S3 bucket
# ---------------------------------------------------------------------------
def cleanup_staging_bucket():
    """Delete the S3 bucket used to stage Lambda code."""
    print(f"🗑️  Deleting staging bucket: {STAGING_BUCKET}...")

    try:
        objects = s3.list_objects_v2(Bucket=STAGING_BUCKET)
        for obj in objects.get("Contents", []):
            s3.delete_object(Bucket=STAGING_BUCKET, Key=obj["Key"])

        s3.delete_bucket(Bucket=STAGING_BUCKET)
        print(f"   ✅ Staging bucket deleted")
    except s3.exceptions.NoSuchBucket:
        print("   Bucket doesn't exist, skipping.")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")


# ---------------------------------------------------------------------------
# Step 7: Delete AgentCore Runtime + ECR repository (Lab 4)
# ---------------------------------------------------------------------------
def cleanup_runtime():
    """Delete the AgentCore Runtime and its ECR container repository.

    LEARNING NOTE: The Runtime and ECR repo are created outside CloudFormation
    by deploy_runtime.py, so they must be deleted manually here.
    The ECR repo contains the Docker image — deleting it with force=True
    removes all image versions inside it.
    """
    print("🗑️  Cleaning up AgentCore Runtime...")
    from lab_helpers.utils import runtime_resource_cleanup
    try:
        runtime_arn = get_ssm_safe("/app/customersupport/agentcore/runtime_arn")
        if runtime_arn:
            runtime_resource_cleanup(runtime_arn)
            print("   ✅ Runtime and ECR repository deleted")
        else:
            print("   No runtime found, skipping.")
    except Exception as e:
        print(f"   ⚠️  Runtime cleanup error: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🧹 Tearing down ALL workshop resources in {REGION}\n")
    print("This will delete everything. Press Ctrl+C within 5 seconds to cancel.\n")

    try:
        for i in range(5, 0, -1):
            print(f"   Starting in {i}...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        exit(0)

    print()
    cleanup_gateway()
    print()
    cleanup_memory()
    print()
    cleanup_cognito()
    print()
    cleanup_ssm_parameters()
    print()
    cleanup_runtime()
    print()
    cleanup_cloudformation()
    print()
    cleanup_staging_bucket()

    print("\n🎉 All workshop resources have been cleaned up!")
    print("Verify in the AWS Console that no unexpected resources remain.")