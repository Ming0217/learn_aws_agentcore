# AWS AgentCore Workshop — Learning Journal

This repo documents my progress through the [AWS AgentCore workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/850fcd5c-fd1f-48d7-932c-ad9babede979/en-US) labs.
Each lab section captures the objectives, what was built, and key takeaways.
The code has been refactored from the original Jupyter notebooks into clean,
production-oriented Python scripts.

---

## Lab 1 — Building a Strands Agent with Tools

### What Was Built

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

## Lab 2 — Persistent Memory with AgentCore Memory

### What Was Built

**`create_memories.py` — Memory Store Setup (one-time script)**
- Creates a named `CustomerSupportMemory` store via `MemoryManager`
- Configures two memory strategies:
  - `UserPreference` — infers and stores customer preferences (e.g., brand loyalty, budget, use case)
  - `Semantic` — extracts factual details from conversations (e.g., owns MacBook Pro, reported overheating)
- Seeds the memory store with previous customer interactions so the agent has history to recall from day one
- Saves the `memory_id` to SSM Parameter Store so `main.py` can reference it without hardcoding

**`main.py` — Agent Runtime (updated)**
- Fetches `memory_id` from SSM at startup — decouples setup from runtime
- Configures `AgentCoreMemoryConfig` with:
  - `session_id` (new UUID per run) for scoping short-term conversation context
  - `actor_id` for namespacing memories per customer
  - `retrieval_config` per namespace with `top_k` and `relevance_score` thresholds
- Passes `AgentCoreMemorySessionManager` into the Strands `Agent` — this hooks into the agent lifecycle to automatically save and retrieve memories without manual API calls
- Updated test queries specifically validate memory recall across sessions

### Key Takeaways

- AgentCore Memory has two layers: short-term (scoped to `session_id`, lost when session ends) and long-term (scoped to `actor_id`, persists across sessions). Personalization relies on long-term memory.
- The `AgentCoreMemorySessionManager` is a "hook" — it intercepts the agent's turn lifecycle to inject retrieved memories before the LLM responds, and persist new interactions after. You don't call memory APIs manually.
- Two memory strategy types serve different purposes: `UserPreference` captures inferred behavioral patterns, `Semantic` captures factual statements. Both are retrieved and injected as context automatically.
- `relevance_score=0.2` is intentionally permissive — for customer support, it's better to over-retrieve and let the LLM filter than to miss relevant context with a high 
threshold.
- Namespacing memories by `{actorId}` is critical for multi-tenant systems — it ensures one customer's memories are never mixed with another's.
- The `create_memories.py` → SSM → `main.py` pattern is the right way to share resource IDs between scripts. It's the same pattern used for the Knowledge Base ID in Lab 1.

---

## Lab 3 — AgentCore Gateway & Secure Tool Sharing

### Original AWS Workshop Objectives
- Centralize reusable tools with AgentCore Gateway
- Add secure authentication with AgentCore Identity (Cognito JWT)
- Expose Lambda functions as MCP-compatible tools
- Connect the agent to shared tools via MCPClient

### What We Built

**`prerequisite/lambda/` — Lambda Function Code (tool backends)**
- `lambda_function.py`: router that dispatches tool calls based on the tool name extracted from Gateway context
- `check_warranty.py`: queries DynamoDB for warranty status by serial number
- `web_search.py`: same DuckDuckGo search from Lab 1, now running in Lambda
- `api_spec.json`: tool schemas that tell the Gateway (and LLM) what tools exist and what parameters they accept

**`deploy_infrastructure.py` — One-Time Infrastructure Deployment**
- Packages Lambda code + DDGS layer, uploads to S3
- Deploys the CloudFormation stack that creates DynamoDB tables, Lambda, IAM roles, S3 buckets, Knowledge Base, and SSM parameters
- Fully automated — one command provisions everything

**`setup_gateway.py` — Gateway Setup**
- Creates Cognito User Pool for JWT authentication
- Creates AgentCore Gateway with MCP protocol and JWT auth
- Adds Lambda as a Gateway target with tool schemas from `api_spec.json`

**`lab_helpers/lab1_strands_agent.py` — Extracted Local Tools**
- Moved `get_product_info`, `get_return_policy`, `get_technical_support`, and `SYSTEM_PROMPT` out of `main.py` into a shared module
- Both Lab 2 and Lab 3 can import the same tools without duplication

**`main.py` — Agent Runtime (updated)**
- Replaced local `web_search` with MCPClient connection to AgentCore Gateway
- Agent now uses a mix of 3 local tools + 2 remote MCP tools (web_search, check_warranty_status)
- `create_agent()` wrapper opens MCP connection, combines local + remote tools, runs the prompt
- JWT token from Cognito is passed in HTTP headers on every Gateway request

**`teardown.py` — Complete Cleanup**
- Deletes Gateway targets + Gateway, Memory store, Cognito resources, SSM parameters
- Deletes CloudFormation stack (handles DynamoDB, Lambda, IAM, S3, KB)
- Deletes staging S3 bucket
- One command to remove everything — no leftover resources or surprise bills

### Key Takeaways

- AgentCore Gateway translates between MCP (what agents speak) and Lambda Invoke (what AWS speaks). The agent never calls Lambda directly.
- The Gateway handles three things the agent shouldn't: protocol translation, JWT authentication, and tool routing to the right backend.
- Tool schemas in `api_spec.json` serve the same purpose as `@tool` docstrings — they tell the LLM what tools exist and how to call them. The difference is they're defined explicitly (JSON) instead of auto-generated from Python type hints.
- Local tools and MCP tools are concatenated into a single list. The LLM doesn't know or care which tools are local vs remote — Strands handles both transparently.
- The `with mcp_client:` context manager ensures the HTTP connection to the Gateway is properly opened and closed per agent invocation.
- CloudFormation makes cleanup trivial — `delete-stack` removes all resources it created in the right dependency order. Resources created outside the stack (Gateway, Memory, Cognito) need separate cleanup, which `teardown.py` handles.

### Setup Order
```bash
python deploy_infrastructure.py   # 1. CloudFormation stack (5-10 min)
python setup_gateway.py            # 2. Gateway + Lambda target
python create_memories.py          # 3. Memory store + seed data
python main.py                     # 4. Run the agent
```

---

*More labs coming soon...*
