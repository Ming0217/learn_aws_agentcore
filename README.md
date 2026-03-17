# AWS AgentCore Workshop — Learning Journal

This repo documents my progress through the AWS AgentCore workshop labs.
Each lab section captures the objectives, what was built, and key takeaways.
The code has been refactored from the original Jupyter notebooks into clean,
production-oriented Python scripts.

---

## Lab 1 — Building a Strands Agent with Tools

### Original AWS Workshop Objectives
- How to create tools using the `@tool` decorator
- How to initialize a Strands agent with model and tools
- How to test your agent locally in a Jupyter notebook

### What We Actually Built (lab_1_cleanup)

Beyond the original lab, this version was refactored for production readiness:

**`main.py` — The Agent Runtime**
- Defined 4 tools using the `@tool` decorator from the Strands Agents framework:
  - `get_return_policy` — mock return policy lookup by product category
  - `get_product_info` — mock product specs lookup
  - `web_search` — live web search via DuckDuckGo (no API key required)
  - `get_technical_support` — RAG-based retrieval from an AWS Bedrock Knowledge Base
- Initialized a `BedrockModel` using Claude Haiku with a low temperature (0.3) for consistent, factual responses
- Wired everything into a Strands `Agent` with a detailed system prompt that guides the LLM on which tool to use for each type of question

**`kb_setup.py` — Knowledge Base Setup (extracted from notebook)**
- Separated one-time ops tasks from agent runtime logic (Single Responsibility Principle)
- `download_files()`: incrementally downloads product support docs from S3, skipping unchanged files using S3 ETag / MD5 comparison
- `sync_knowledge_base()`: triggers a Bedrock ingestion job to index S3 documents into the Knowledge Base, with async polling for job completion

### Key Takeaways

- The `@tool` decorator uses the function's docstring and type hints to generate the JSON schema sent to the LLM — clear docstrings directly influence how well the agent picks the right tool
- The Strands agentic loop (ReAct pattern) handles the tool-calling cycle automatically: LLM reasons → selects tool → executes → reasons again → responds
- `temperature=0.3` on the model keeps customer support responses factual and consistent; higher values are better suited for creative tasks
- Bedrock Knowledge Base ingestion is incremental by default — no need to manually diff documents before syncing
- SSM Parameter Store is the right place for resource IDs (KB IDs, data source IDs) — never hardcode them
- Jupyter notebooks are great for labs but not for production; separating concerns into focused scripts makes code testable, reusable, and easier to evolve (e.g., `kb_setup.py` could become a Lambda triggered by S3 events)

---

*More labs coming soon...*
