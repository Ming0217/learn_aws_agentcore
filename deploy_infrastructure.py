"""
deploy_infrastructure.py - One-Time Infrastructure Deployment
==============================================================
Packages Lambda code, uploads to S3, and deploys the CloudFormation stack
that provisions all AWS resources needed for the workshop.

WHAT THIS CREATES:
  - DynamoDB tables (warranty data, customer profiles) with synthetic data
  - Lambda function (check_warranty_status + web_search handlers)
  - Lambda DDGS layer (DuckDuckGo dependency)
  - IAM roles (Gateway role, Runtime role, Lambda execution role)
  - S3 bucket for Knowledge Base documents
  - Bedrock Knowledge Base + data source + vector store
  - SSM parameters for all resource IDs

CLEANUP: Run teardown.py to delete everything, or:
  aws cloudformation delete-stack --stack-name agentcore-workshop

USAGE:
  python deploy_infrastructure.py
"""

import boto3
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STACK_NAME = "agentcore-workshop"
CFN_TEMPLATE_PATH = "prerequisite/infrastructure.yaml"
LAMBDA_SOURCE_DIR = "prerequisite/lambda/python"

boto_session = boto3.Session()
REGION = boto_session.region_name
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

# S3 bucket to stage Lambda code — CloudFormation needs it in S3
STAGING_BUCKET = f"{ACCOUNT_ID}-{REGION}-agentcore-staging"

s3 = boto3.client("s3")
cfn = boto3.client("cloudformation")


# ---------------------------------------------------------------------------
# Step 1: Create the staging S3 bucket
# ---------------------------------------------------------------------------
def create_staging_bucket():
    """Create an S3 bucket to stage Lambda code for CloudFormation.

    LEARNING NOTE: CloudFormation can't deploy Lambda code from your local
    filesystem. The code must be in S3 first. This staging bucket is a
    temporary holding area — it's only needed during deployment and can be
    deleted afterward (teardown.py handles this).

    The us-east-1 region is special — it doesn't accept LocationConstraint
    in create_bucket. Every other region requires it. This is a well-known
    AWS quirk you'll encounter in many projects.
    """
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=STAGING_BUCKET)
        else:
            s3.create_bucket(
                Bucket=STAGING_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        print(f"✅ Created staging bucket: {STAGING_BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ Staging bucket already exists: {STAGING_BUCKET}")
    except Exception as e:
        print(f"❌ Error creating staging bucket: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2: Package and upload Lambda code
# ---------------------------------------------------------------------------
def package_and_upload_lambda():
    """
    Zip the Lambda source files and upload to S3.

    LEARNING NOTE: CloudFormation's AWS::Lambda::Function resource can't
    inline code larger than 4KB. For real Lambda functions, you package
    the code as a zip, upload to S3, and reference it in the template
    via S3Bucket/S3Key parameters.
    """
    lambda_zip_key = "lambda/customer-support-lambda.zip"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in os.listdir(LAMBDA_SOURCE_DIR):
                filepath = os.path.join(LAMBDA_SOURCE_DIR, filename)
                if os.path.isfile(filepath) and filename.endswith(".py"):
                    zf.write(filepath, filename)
                    print(f"   Added: {filename}")

        s3.upload_file(zip_path, STAGING_BUCKET, lambda_zip_key)
        print(f"✅ Lambda code uploaded to s3://{STAGING_BUCKET}/{lambda_zip_key}")
    finally:
        os.unlink(zip_path)

    return lambda_zip_key


# ---------------------------------------------------------------------------
# Step 3: Package and upload DDGS Lambda layer
# ---------------------------------------------------------------------------
def package_and_upload_ddgs_layer():
    """
    Install the ddgs package into a Lambda layer structure and upload to S3.

    LEARNING NOTE: Lambda layers let you share dependencies across functions.
    The directory structure must be python/ at the root of the zip — Lambda
    adds this to the Python path automatically.
    """
    layer_zip_key = "lambda/ddgs-layer.zip"

    with tempfile.TemporaryDirectory() as tmpdir:
        python_dir = os.path.join(tmpdir, "python")
        os.makedirs(python_dir)

        print("   Installing ddgs package...")
        subprocess.check_call(
            ["uv", "pip", "install", "ddgs",
             "--target", python_dir, "--quiet"]
        )

        zip_path = os.path.join(tmpdir, "ddgs-layer.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(python_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    arcname = os.path.relpath(full_path, tmpdir)
                    zf.write(full_path, arcname)

        s3.upload_file(zip_path, STAGING_BUCKET, layer_zip_key)
        print(f"✅ DDGS layer uploaded to s3://{STAGING_BUCKET}/{layer_zip_key}")

    return layer_zip_key


# ---------------------------------------------------------------------------
# Step 4: Deploy CloudFormation stack
# ---------------------------------------------------------------------------
def deploy_stack(lambda_zip_key, layer_zip_key):
    """
    Deploy the CloudFormation stack with all workshop infrastructure.

    LEARNING NOTE: CAPABILITY_IAM is required because the template creates
    IAM roles. Without this acknowledgment, CloudFormation refuses to deploy.
    """
    # LEARNING NOTE: CloudFormation has a 51,200 byte limit for inline templates
    # (TemplateBody). Our annotated template exceeds this, so we upload it to S3
    # and use TemplateURL instead. This is the standard approach for larger templates.
    template_s3_key = "cloudformation/infrastructure.yaml"
    s3.upload_file(CFN_TEMPLATE_PATH, STAGING_BUCKET, template_s3_key)
    template_url = f"https://{STAGING_BUCKET}.s3.amazonaws.com/{template_s3_key}"
    print(f"   Template uploaded to s3://{STAGING_BUCKET}/{template_s3_key}")

    params = [
        {"ParameterKey": "LambdaS3Bucket", "ParameterValue": STAGING_BUCKET},
        {"ParameterKey": "LambdaS3Key", "ParameterValue": lambda_zip_key},
        {"ParameterKey": "LayerS3Key", "ParameterValue": layer_zip_key},
    ]

    try:
        # Check if stack already exists
        try:
            cfn.describe_stacks(StackName=STACK_NAME)
            print(f"Stack {STACK_NAME} already exists. Updating...")
            cfn.update_stack(
                StackName=STACK_NAME,
                TemplateURL=template_url,
                Parameters=params,
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
            )
        except cfn.exceptions.ClientError as e:
            if "does not exist" in str(e):
                print(f"Creating stack: {STACK_NAME}")
                cfn.create_stack(
                    StackName=STACK_NAME,
                    TemplateURL=template_url,
                    Parameters=params,
                    Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
                )
            elif "No updates are to be performed" in str(e):
                print("✅ Stack is already up to date.")
                return
            else:
                raise

        # Wait for stack to complete
        print("⏳ Waiting for stack deployment (this may take 5-10 minutes)...")
        waiter = cfn.get_waiter("stack_create_complete")
        try:
            waiter.wait(
                StackName=STACK_NAME,
                WaiterConfig={"Delay": 15, "MaxAttempts": 60},
            )
        except Exception:
            # Try update waiter if create waiter fails (stack was updating)
            waiter = cfn.get_waiter("stack_update_complete")
            waiter.wait(
                StackName=STACK_NAME,
                WaiterConfig={"Delay": 15, "MaxAttempts": 60},
            )

        # Print outputs
        stack = cfn.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
        print(f"\n✅ Stack {STACK_NAME} deployed successfully!")
        print("\nStack outputs:")
        for output in stack.get("Outputs", []):
            print(f"   {output['OutputKey']}: {output['OutputValue']}")

    except Exception as e:
        print(f"❌ Stack deployment failed: {e}")
        print("Check the CloudFormation console for detailed error messages.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Deploying workshop infrastructure to {REGION} (account: {ACCOUNT_ID})\n")

    print("=== Step 1: Creating staging bucket ===")
    create_staging_bucket()

    print("\n=== Step 2: Packaging Lambda code ===")
    lambda_zip_key = package_and_upload_lambda()

    print("\n=== Step 3: Packaging DDGS layer ===")
    layer_zip_key = package_and_upload_ddgs_layer()

    print(f"\n=== Step 4: Deploying CloudFormation stack ({STACK_NAME}) ===")
    deploy_stack(lambda_zip_key, layer_zip_key)

    print("\n🎉 Infrastructure deployment complete!")
    print("You can now run: python setup_gateway.py")
