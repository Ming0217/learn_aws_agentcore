"""
web_search.py - Web Search Tool (Lambda version)
==================================================
This is the same web_search function from Lab 1, now packaged as a Lambda
handler for use behind AgentCore Gateway.

LEARNING NOTE: The code is identical to what was in main.py. The only
difference is WHERE it runs — in a Lambda instead of in-process with the agent.
This is the key benefit of Gateway: you can take existing tool code and
centralize it without rewriting the logic.
"""

from ddgs import DDGS


def web_search(keywords: str, region: str = "us-en", max_results: int = 5) -> str:
    """Search the web for updated information.

    Args:
        keywords (str): The search query keywords.
        region (str): The search region: wt-wt, us-en, uk-en, ru-ru, etc.
        max_results (int): The maximum number of results to return.

    Returns:
        List of dictionaries with search results.
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return results if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"
