"""
lambda_function.py - AgentCore Gateway Lambda Handler
======================================================
This Lambda function serves as the backend for AgentCore Gateway.
It handles multiple tools (check_warranty_status, web_search) routed
by the Gateway based on the tool name.

THE LAMBDA'S ROLE IN THE CHAIN:
  MCPClient ──HTTP+JWT──→ Gateway ──Lambda Invoke──→ Lambda
                                                     (you are here)

  The Lambda is the "kitchen" — it does the actual work. It doesn't know
  about MCP, JWT tokens, or the agent. It just receives:
    - Tool parameters in the `event` dict (e.g., {"serial_number": "ABC123"})
    - Tool name in `context.client_context.custom["bedrockAgentCoreToolName"]`
  And returns a result dict with statusCode and body.

  The Gateway handles all the protocol translation and auth BEFORE the
  Lambda is invoked. By the time the Lambda runs, the request is already
  authenticated and the parameters are already extracted.

HOW ROUTING WORKS:
  AgentCore Gateway invokes this Lambda and passes the tool name in the
  Lambda context (not the event). The tool name is namespaced as:
    "TargetName___tool_name"
  We split on "___" to extract just the tool name, then route to the
  right handler function.

  Tool parameters are passed in the Lambda event as a flat dict.
"""

from check_warranty import check_warranty_status
from web_search import web_search


def get_named_parameter(event, name):
    """Extract a named parameter from the Lambda event."""
    if name not in event:
        return None
    return event.get(name)


def lambda_handler(event, context):
    """
    Main Lambda entry point. Routes tool calls to the appropriate handler.

    LEARNING NOTE: This is a mini-router pattern — one Lambda handles multiple
    tools. This is more cost-effective than one Lambda per tool, and simpler
    to manage. The Gateway handles the MCP protocol translation; this Lambda
    just needs to do the actual work.
    """
    print(f"Event: {event}")
    print(f"Context: {context}")

    # Extract the tool name from the Gateway-provided context.
    # Format: "TargetName___tool_name" → we want just "tool_name"
    extended_tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
    resource = extended_tool_name.split("___")[1]

    print(resource)

    if resource == "check_warranty_status":
        serial_number = get_named_parameter(event=event, name="serial_number")
        customer_email = get_named_parameter(event=event, name="customer_email")

        if not serial_number:
            return {"statusCode": 400, "body": "❌ Please provide serial_number"}

        try:
            warranty_status = check_warranty_status(
                serial_number=serial_number, customer_email=customer_email
            )
        except Exception as e:
            print(e)
            return {"statusCode": 400, "body": f"❌ {e}"}

        return {"statusCode": 200, "body": warranty_status}

    elif resource == "web_search":
        keywords = get_named_parameter(event=event, name="keywords")
        region = get_named_parameter(event=event, name="region") or "us-en"
        max_results = get_named_parameter(event=event, name="max_results") or 5

        if not keywords:
            return {"statusCode": 400, "body": "❌ Please provide keywords for search"}

        try:
            search_results = web_search(
                keywords=keywords, region=region, max_results=int(max_results)
            )
        except Exception as e:
            print(e)
            return {"statusCode": 400, "body": f"❌ {e}"}

        return {"statusCode": 200, "body": f"🔍 Search Results: {search_results}"}

    return {"statusCode": 400, "body": f"❌ Unknown toolname: {resource}"}
