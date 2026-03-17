"""
kb_setup.py - Knowledge Base Setup Script
==========================================
This script handles the ONE-TIME (or on-demand) setup tasks for the Bedrock Knowledge Base.
It is intentionally separated from main.py because:
  - It's an ops/infrastructure task, not agent runtime logic
  - It only needs to run when new product docs are added or the KB needs resyncing
  - Keeping it separate follows the Single Responsibility Principle

In production, this could be:
  - A Lambda function triggered by S3 PutObject events (new docs uploaded → auto-sync)
  - A Step Functions workflow for more complex ingestion pipelines
  - A scheduled job (e.g., nightly sync via EventBridge)

AWS Services used:
  - S3: Object storage for raw product technical support documents
  - SSM Parameter Store: Secure storage for KB IDs (avoids hardcoding resource IDs)
  - Bedrock Agent (Knowledge Base): Managed RAG (Retrieval-Augmented Generation) service
    that indexes your documents so the agent can search them semantically
"""

import os
import hashlib
import time
import boto3


# ---------------------------------------------------------------------------
# STEP 1: Download product technical support files from S3 (incremental)
# ---------------------------------------------------------------------------
# WHY: The Bedrock Knowledge Base needs source documents to index.
#      These are stored in S3 (e.g., product manuals, troubleshooting guides).
#      We download them locally here for inspection/debugging purposes.
#      In production, Bedrock can ingest directly from S3 — no local download needed.
#
# INCREMENTAL DOWNLOAD STRATEGY:
#   Rather than re-downloading everything on every run, we compare S3 objects
#   against what's already on disk using two checks:
#     1. Does the file exist locally at all? (new file check)
#     2. Does the local file's MD5 match the S3 ETag? (changed file check)
#
#   LEARNING NOTE: S3 ETags are MD5 checksums for single-part uploads.
#   For multipart uploads, the ETag format is different (hash-of-hashes with a suffix
#   like "-2"), so we only use ETag comparison for single-part files.
#   A safer alternative for large files is to compare S3 LastModified timestamp
#   vs local file mtime, which works regardless of upload method.
# ---------------------------------------------------------------------------
def get_local_md5(file_path: str) -> str:
    """Compute the MD5 checksum of a local file for comparison with S3 ETags."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_files():
    """
    Incrementally downloads product technical support files from S3.
    Skips files that already exist locally and are unchanged (via ETag/MD5 check).
    Only downloads new or modified files.

    The bucket name follows the convention: {account_id}-{region}-kb-data-bucket
    This naming pattern is common in AWS labs to ensure globally unique bucket names.
    """
    # Resolve the AWS account ID and region dynamically.
    # LEARNING NOTE: Never hardcode account IDs or regions — always resolve at runtime.
    # boto3.client("sts").get_caller_identity() is the standard way to get your account ID.
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = boto3.Session().region_name
    bucket_name = f"{account_id}-{region}-kb-data-bucket"

    # Create the local destination folder if it doesn't exist.
    # exist_ok=True prevents an error if the folder already exists.
    os.makedirs("knowledge_base_data", exist_ok=True)

    # List all objects in the S3 bucket.
    # LEARNING NOTE: list_objects_v2 is the modern API (v1 is deprecated).
    # For buckets with >1000 objects, you'd need to handle pagination via 'NextContinuationToken'.
    s3 = boto3.client("s3")
    objects = s3.list_objects_v2(Bucket=bucket_name)

    downloaded, skipped, updated = [], [], []

    for obj in objects.get("Contents", []):
        file_name = obj["Key"]
        local_path = f"knowledge_base_data/{file_name}"
        s3_etag = obj["ETag"].strip('"')  # ETags come wrapped in quotes

        if os.path.exists(local_path):
            # LEARNING NOTE: S3 ETags for multipart uploads contain a "-N" suffix
            # and are NOT a simple MD5. We detect this and fall back to a
            # size comparison to avoid false "file changed" detections.
            is_multipart = "-" in s3_etag
            if is_multipart:
                # Fallback: compare file size as a lightweight change check
                local_size = os.path.getsize(local_path)
                s3_size = obj["Size"]
                is_changed = local_size != s3_size
            else:
                # Single-part upload: safe to compare MD5 vs ETag directly
                is_changed = get_local_md5(local_path) != s3_etag

            if not is_changed:
                print(f"⏭️  Skipped (unchanged): {file_name}")
                skipped.append(file_name)
                continue
            else:
                print(f"🔄 Updating (changed): {file_name}")
                updated.append(file_name)
        else:
            print(f"⬇️  Downloading (new): {file_name}")
            downloaded.append(file_name)

        s3.download_file(bucket_name, file_name, local_path)

    # Summary report
    print(f"\n✅ Download complete.")
    print(f"   New: {len(downloaded)} | Updated: {len(updated)} | Skipped: {len(skipped)}")
    if downloaded:
        print(f"   New files: {', '.join(downloaded)}")
    if updated:
        print(f"   Updated files: {', '.join(updated)}")


# ---------------------------------------------------------------------------
# STEP 2: Sync the Bedrock Knowledge Base with the S3 documents
# ---------------------------------------------------------------------------
# WHY: After uploading new/updated documents to S3, the Knowledge Base index
#      is NOT automatically updated. You must trigger an ingestion job to
#      re-index the documents so the agent can retrieve the latest content.
#
# HOW IT WORKS (RAG pipeline):
#   S3 docs → Bedrock chunks & embeds text → stores vectors in a vector store
#   → agent queries vector store at runtime to find relevant passages
# ---------------------------------------------------------------------------
def sync_knowledge_base():
    """
    Triggers a Bedrock Knowledge Base ingestion job to index documents from S3,
    then polls until the job completes.

    LEARNING NOTE: Ingestion jobs are asynchronous — you start them and poll for status.
    This is a common AWS pattern for long-running operations (also seen in Glue, EMR, etc.)
    """
    # Initialize AWS clients
    ssm = boto3.client("ssm")
    bedrock = boto3.client("bedrock-agent")
    s3 = boto3.client("s3")

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = boto3.Session().region_name

    # Retrieve the Knowledge Base ID and Data Source ID from SSM Parameter Store.
    # LEARNING NOTE: SSM Parameter Store is the right place to store resource IDs like these.
    # It avoids hardcoding IDs that change between environments (dev/staging/prod).
    # The parameter path uses a namespacing convention: /{account_id}-{region}/kb/...
    kb_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/knowledge-base-id")["Parameter"]["Value"]
    ds_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/data-source-id")["Parameter"]["Value"]

    # List the files currently in S3 so we can report what was ingested.
    bucket_name = f"{account_id}-{region}-kb-data-bucket"
    s3_objects = s3.list_objects_v2(Bucket=bucket_name)
    file_names = [obj["Key"] for obj in s3_objects.get("Contents", [])]

    # Start the ingestion job.
    # LEARNING NOTE: A Data Source in Bedrock KB represents the S3 location of your documents.
    # The ingestion job reads from that data source, chunks the text, generates embeddings
    # (vector representations), and stores them in the configured vector store (e.g., OpenSearch).
    #
    # INCREMENTAL SYNC — NO CUSTOM DIFFING NEEDED:
    #   Unlike the download step, you do NOT need to manually track which files are new
    #   or changed here. Bedrock's ingestion job handles this natively:
    #     - It tracks S3 object metadata (ETag, LastModified) from the previous sync
    #     - Only re-chunks and re-embeds documents that are new or modified
    #     - Removes vectors for documents deleted from S3
    #   This makes repeated calls to start_ingestion_job safe and efficient — it's idempotent.
    response = bedrock.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description="Sync product technical support documents"
    )

    job_id = response["ingestionJob"]["ingestionJobId"]
    print(f"Bedrock KB ingestion job started. Job ID: {job_id}")

    # Poll the job status until it reaches a terminal state (COMPLETE or FAILED).
    # LEARNING NOTE: Polling with time.sleep() is simple but not ideal for production.
    # Better alternatives: EventBridge events, Step Functions wait states, or SNS notifications.
    while True:
        job = bedrock.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id
        )["ingestionJob"]

        status = job["status"]
        print(f"Job status: {status}...")

        if status in ["COMPLETE", "FAILED"]:
            break

        time.sleep(10)  # Wait 10 seconds before polling again

    # Report the final result
    if status == "COMPLETE":
        file_count = job.get("statistics", {}).get("numberOfDocumentsScanned", 0)
        files_list = ", ".join(file_names)
        print(f"✅ KB sync complete. Ingested {file_count} files.")
        print(f"Files ingested: {files_list}")
    else:
        print(f"❌ KB sync failed with status: {status}")
        print(f"Failure reasons: {job.get('failureReasons', 'No details available')}")


# ---------------------------------------------------------------------------
# Entrypoint: Run both steps in sequence
# ---------------------------------------------------------------------------
# LEARNING NOTE: The `if __name__ == "__main__"` guard ensures this only runs
# when the script is executed directly (e.g., `python kb_setup.py`),
# NOT when it's imported as a module by another script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Step 1: Downloading files from S3 ===")
    download_files()

    print("\n=== Step 2: Syncing Bedrock Knowledge Base ===")
    sync_knowledge_base()
