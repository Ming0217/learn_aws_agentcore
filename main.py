# Import libraries
import boto3
from boto3.session import Session

from ddgs.exceptions import DDGSException, RatelimitException
from ddgs import DDGS

from strands.tools import tool

# Get boto session
boto_session = Session()
region = boto_session.region_name

@tool
def get_return_policy(product_category: str) -> str:
    """
    Get return policy information for a specific product category.

    Args:
        product_category: Electronics category (e.g., 'smartphones', 'laptops', 'accessories')

    Returns:
        Formatted return policy details including timeframes and conditions
    """
    # Mock return policy database - in real implementation, this would query policy database
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

    # Default policy for unlisted categories
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

@tool
def get_product_info(product_type: str) -> str:
    """
    Get detailed technical specifications and information for electronics products.

    Args:
        product_type: Electronics product type (e.g., 'laptops', 'smartphones', 'headphones', 'monitors')
    Returns:
        Formatted product information including warranty, features, and policies
    """
    # Mock product catalog - in real implementation, this would query a product database
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
        return "Rate limit reached. Please try again later."
    except DDGSException as e:
        return f"Search error: {e}"
    except Exception as e:
        return f"Search error: {str(e)}"


print("✅ Web search tool ready")


# import os


# def download_files():
#     # Get account and region
#     account_id = boto3.client("sts").get_caller_identity()["Account"]
#     region = boto3.Session().region_name
#     bucket_name = f"{account_id}-{region}-kb-data-bucket"

#     # Create local folder
#     os.makedirs("knowledge_base_data", exist_ok=True)

#     # Download all files
#     s3 = boto3.client("s3")
#     objects = s3.list_objects_v2(Bucket=bucket_name)

#     for obj in objects["Contents"]:
#         file_name = obj["Key"]
#         s3.download_file(bucket_name, file_name, f"knowledge_base_data/{file_name}")
#         print(f"Downloaded: {file_name}")

#     print("All files saved to: knowledge_base_data/")


# # Run it
# download_files()

# import time

# # Get parameters
# ssm = boto3.client("ssm")
# bedrock = boto3.client("bedrock-agent")
# s3 = boto3.client("s3")

# account_id = boto3.client("sts").get_caller_identity()["Account"]
# region = boto3.Session().region_name

# kb_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/knowledge-base-id")[
#     "Parameter"
# ]["Value"]
# ds_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/data-source-id")[
#     "Parameter"
# ]["Value"]

# # Get file names from S3 bucket
# bucket_name = f"{account_id}-{region}-kb-data-bucket"
# s3_objects = s3.list_objects_v2(Bucket=bucket_name)
# file_names = [obj["Key"] for obj in s3_objects.get("Contents", [])]

# # Start sync job
# response = bedrock.start_ingestion_job(
#     knowledgeBaseId=kb_id, dataSourceId=ds_id, description="Quick sync"
# )

# job_id = response["ingestionJob"]["ingestionJobId"]
# print("Bedrock knowledge base sync job started, ingesting the data files from s3")

# # Monitor until complete
# while True:
#     job = bedrock.get_ingestion_job(
#         knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
#     )["ingestionJob"]

#     status = job["status"]

#     if status in ["COMPLETE", "FAILED"]:
#         break

#     time.sleep(10)

# # Print final result
# if status == "COMPLETE":
#     file_count = job.get("statistics", {}).get("numberOfDocumentsScanned", 0)
#     files_list = ", ".join(file_names)
#     print(
#         f"Bedrock knowledge base sync job completed Successfully, ingested {file_count} files"
#     )
#     print(f"Files ingested: {files_list}")
# else:
#     print(f"Bedrock knowledge base sync job failed with status: {status}")


from strands.models import BedrockModel
from strands import Agent
from strands_tools import retrieve


@tool
def get_technical_support(issue_description: str) -> str:
    try:
        # Get KB ID from parameter store
        ssm = boto3.client("ssm")
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        region = boto3.Session().region_name

        kb_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/knowledge-base-id")[
            "Parameter"
        ]["Value"]
        print(f"Successfully retrieved KB ID: {kb_id}")

        # Use strands retrieve tool
        tool_use = {
            "toolUseId": "tech_support_query",
            "input": {
                "text": issue_description,
                "knowledgeBaseId": kb_id,
                "region": region,
                "numberOfResults": 3,
                "score": 0.4,
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
3. web_search() - To access current technical documentation, or for updated information. 
4. get_technical_support() - For troubleshooting issues, setup guides, maintenance tips, and detailed technical assistance
For any technical problems, setup questions, or maintenance concerns, always use the get_technical_support() tool as it contains our comprehensive technical documentation and step-by-step guides.

Always use the appropriate tool to get accurate, up-to-date information rather than making assumptions about electronic products or specifications."""

# Initialize the Bedrock model (Anthropic Claude 3.7 Sonnet)
model = BedrockModel(
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature=0.3,
    region_name=region,
)

# Create the customer support agent with all tools
agent = Agent(
    model=model,
    tools=[
        get_product_info,  # Tool 1: Simple product information lookup
        get_return_policy,  # Tool 2: Simple return policy lookup
        web_search,  # Tool 3: Access the web for updated information
        get_technical_support,  # Tool 4: Technical support & troubleshooting
    ],
    system_prompt=SYSTEM_PROMPT,
)

print("Customer Support Agent created successfully!")

response = agent("What's the return policy for my thinkpad X1 Carbon?")

response = agent(
    "I bought an iphone 14 last month. I don't like it because it heats up. How do I solve it?"
)