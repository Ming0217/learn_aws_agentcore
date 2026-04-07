"""
lab_helpers/lab1_strands_agent.py - Reusable Local Tools & System Prompt
=========================================================================
Extracted from main.py so that both Lab 2 (main.py) and Lab 3 (setup_gateway.py)
can import the same tools without code duplication.

LEARNING NOTE: This is exactly the problem Lab 3 solves at a larger scale.
For tools that are specific to this agent (get_product_info, get_return_policy,
get_technical_support), keeping them as local @tool functions is fine.
For tools that multiple agents need (web_search, check_warranty_status),
they belong behind AgentCore Gateway.

These 3 tools remain local because they're specific to the customer support
use case and don't need to be shared with other agents.
"""

import boto3
from boto3.session import Session
from strands.tools import tool
from strands_tools import retrieve

boto_session = Session()
region = boto_session.region_name

# Model ID used by both local agent (main.py) and runtime agent (lab4_runtime.py).
# Centralised here so changing the model only requires one edit.
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


@tool
def get_return_policy(product_category: str) -> str:
    """
    Get return policy information for a specific product category.

    Args:
        product_category: Electronics category (e.g., 'smartphones', 'laptops', 'accessories')

    Returns:
        Formatted return policy details including timeframes and conditions
    """
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


@tool
def get_product_info(product_type: str) -> str:
    """
    Get detailed technical specifications and information for electronics products.

    Args:
        product_type: Electronics product type (e.g., 'laptops', 'smartphones', 'headphones', 'monitors')
    Returns:
        Formatted product information including warranty, features, and policies
    """
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
        return f"Technical specifications for {product_type} not available. Please contact our technical support team."

    return (
        f"Technical Information - {product_type.title()}:\n\n"
        f"• Warranty: {product['warranty']}\n"
        f"• Specifications: {product['specs']}\n"
        f"• Key Features: {product['features']}\n"
        f"• Compatibility: {product['compatibility']}\n"
        f"• Support: {product['support']}"
    )


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
        rgn = boto3.Session().region_name

        kb_id = ssm.get_parameter(
            Name=f"/{account_id}-{rgn}/kb/knowledge-base-id"
        )["Parameter"]["Value"]
        print(f"Successfully retrieved KB ID: {kb_id}")

        tool_use = {
            "toolUseId": "tech_support_query",
            "input": {
                "text": issue_description,
                "knowledgeBaseId": kb_id,
                "region": rgn,
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


# ---------------------------------------------------------------------------
# System Prompt — shared across Lab 2 and Lab 3
# ---------------------------------------------------------------------------
# LEARNING NOTE: In Lab 3, the system prompt is updated to include the new
# MCP-based tools (check_warranty_status, web_search via Gateway).
# This version is the base prompt; main.py may extend it.
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
5. check_warranty_status() - To check warranty status using a product serial number

For any technical problems, setup questions, or maintenance concerns, always use the get_technical_support() tool
as it contains our comprehensive technical documentation and step-by-step guides.

Always use the appropriate tool to get accurate, up-to-date information rather than making assumptions about
electronic products or specifications."""
