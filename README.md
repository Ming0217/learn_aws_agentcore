# AWS AgentCore Workshop — Learning Journal

This repo documents my progress through the [AWS AgentCore workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/850fcd5c-fd1f-48d7-932c-ad9babede979/en-US) labs.
Each lab section captures the objectives, what was built, and key takeaways.
The code has been refactored from the original Jupyter notebooks into clean,
production-oriented Python scripts.

---

## Prerequisites

### Local Environment
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — Python package manager (used instead of pip)
- AWS CLI v2 configured with valid credentials (`aws configure`)
- Git

### AWS Account Access
- An AWS account with permissions to create: IAM roles, DynamoDB tables, Lambda functions, S3 buckets, SSM parameters, Bedrock Knowledge Bases, AgentCore resources (Gateway, Memory, Runtime), Cognito User Pools, Secrets Manager secrets
- Amazon Bedrock model access enabled for:
  - `anthropic.claude-haiku-4-5-20251001-v1:0` (or your chosen model)
  - `amazon.titan-embed-text-v2:0` (used by the Knowledge Base for embeddings)

### Python Dependencies (managed by uv)
Defined in `pyproject.toml`, installed via `uv sync`:
- `strands-agents` — Strands Agents framework (agentic loop, tool management)
- `strands-agents-tools` — built-in tools for Strands (includes `retrieve` for KB)
- `boto3` / `botocore` — AWS SDK for Python
- `bedrock-agentcore` — AgentCore client library (Memory, Gateway integration)
- `bedrock-agentcore-starter-toolkit` — helper utilities for AgentCore operations
- `aws-opentelemetry-distro` — OpenTelemetry for observability (Lab 4)
- `ddgs` — DuckDuckGo search (used by web_search tool)
- `pyyaml` — YAML parsing (used by config utilities)

### AWS Infrastructure (provisioned by CloudFormation)
The workshop's `CustomerSupportStackInfra` CloudFormation stack (or our `deploy_infrastructure.py`) creates:
- 2 DynamoDB tables — `WarrantyTable` (serial_number → warranty info) and `CustomerProfileTable` (customer_id → profile)
- 1 Lambda function — `CustomerSupportLambda` with `check_warranty_status` and `web_search` handlers
- 1 Lambda layer — DDGS package for DuckDuckGo search in Lambda
- 1 S3 bucket — `{account}-{region}-kb-data-bucket` for Knowledge Base documents
- 1 Bedrock Knowledge Base — indexes technical support docs from S3 using Titan embeddings + S3 Vectors
- 6 IAM roles — for Gateway, Runtime, Lambda execution, Bedrock service, data seeding, and KB setup
- 5+ SSM parameters — storing table names, Lambda ARN, IAM role ARNs, KB/DS IDs

### Resources Created by Setup Scripts (outside CloudFormation)
These are created by running the setup scripts and must be cleaned up separately via `teardown.py`:
- AgentCore Gateway — MCP server with Cognito JWT auth (`setup_gateway.py`)
- AgentCore Memory store — persistent customer memory with UserPreference + Semantic strategies (`create_memories.py`)
- Cognito User Pool + App Client — issues JWT tokens for Gateway auth (`get_or_create_cognito_pool()`)
- Secrets Manager secret — stores Cognito config (pool_id, client_id, client_secret)
- Additional SSM parameters — gateway_id, gateway_url, memory_id, cognito config

### Quick Start
```bash
# 1. Clone and install dependencies
git clone <your-repo-url>
cd learn_aws_agentcore
uv sync

# 2. Provision AWS infrastructure (skip if using workshop-provided stack)
python deploy_infrastructure.py

# 3. Set up AgentCore Gateway
python setup_gateway.py

# 4. Set up AgentCore Memory and seed customer history
python create_memories.py

# 5. Run the agent
python main.py

# 6. Clean up everything when done
python teardown.py
```

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

---

## Lab 4 — Deploy to Production with AgentCore Runtime

### Original AWS Workshop Objectives
- Deploy the agent to AgentCore Runtime for production-ready hosting
- Add comprehensive observability with CloudWatch GenAI Observability
- Demonstrate session continuity and memory persistence in production
- Implement JWT token propagation from Runtime to Gateway

### What We Built

**Deployment Flow Diagram**

```
YOUR CODE                    STARTER TOOLKIT              AWS
─────────────────────────    ────────────────────────     ──────────────────────────────

lab_helpers/
  lab4_runtime.py  ──────→  deploy_runtime.py
  (agent logic +             │
   @app.entrypoint)          │  configure()
                             │  ├── reads lab4_runtime.py
requirements.txt  ──────────→│  ├── reads requirements.txt
  (dependencies)             │  ├── generates → Dockerfile
                             │  └── generates → .bedrock_agentcore.yaml
                             │
                             │  launch()
                             │  ├── builds Docker image (via CodeBuild)
                             │  ├── pushes image ──────────────────────→  ECR
                             │  └── deploys container ─────────────────→  AgentCore Runtime
                             │                                              (always-on HTTPS endpoint)
                             │
                             │  saves agent ARN ──────────────────────→  SSM Parameter Store
                             │
invoke_runtime.py  ─────────→  runtime.configure()  (restores state from .bedrock_agentcore.yaml)
  (test queries)              runtime.invoke()
                             │  ├── sends JWT + payload via HTTPS
                             │  └── ─────────────────────────────────→  AgentCore Runtime
                             │                                              │
                             │                                              │  validates JWT (Cognito)
                             │                                              │  calls @app.entrypoint
                             │                                              │  propagates JWT to Gateway
                             │                                              ↓
                             │                                           AgentCore Gateway
                             │                                              │
                             │                                              ↓
                             │                                           Lambda (tools)
                             │                                              │
                             ←──────────────────────────────────────────── response
```

**File Roles**

```
lab_helpers/lab4_runtime.py   → the agent code that runs INSIDE the container
requirements.txt              → tells Docker what packages to install in the container
Dockerfile                    → generated by configure(), defines the container image
.bedrock_agentcore.yaml       → generated by configure(), stores deployment config
                                (agent name, ARN, ECR repo, auth config, region)
deploy_runtime.py             → one-time script: configure + launch + wait for READY
invoke_runtime.py             → test script: reconnects to deployed Runtime and runs queries
```

**`lab_helpers/lab4_runtime.py` — Runtime-Ready Agent**
- Wraps the same agent logic from Labs 1-3 inside `BedrockAgentCoreApp`
- Only 4 lines of code added to make the agent production-ready:
  1. `from bedrock_agentcore.runtime import BedrockAgentCoreApp`
  2. `app = BedrockAgentCoreApp()`
  3. `@app.entrypoint` decorator on the handler function
  4. `app.run()` at the bottom
- The `@app.entrypoint` function receives the JWT token via `context.request_headers` and propagates it to the Gateway — same token authenticates both the Runtime call and the Gateway tool calls
- `MEMORY_ID` is passed as an environment variable (not SSM) for container startup

**`requirements.txt` — Container Dependencies**
- Lists Python packages for the Docker image built by the starter toolkit
- Separate from `pyproject.toml` — the container only needs runtime deps, not dev tools

**`deploy_runtime.py` — One-Time Deployment Script**
- Creates the IAM execution role for the Runtime container
- Calls `agentcore_runtime.configure()` to generate Dockerfile + `.bedrock_agentcore.yaml`
- Calls `agentcore_runtime.launch()` to trigger the CodeBuild pipeline (build → ECR push → deploy)
- Polls status until `READY` or `FAILED`
- Saves Runtime ARN to SSM

**`invoke_runtime.py` — Test the Deployed Agent**
- Demonstrates 4 scenarios: tool listing, session continuity, warranty check (MCP), memory recall
- Uses `agentcore_runtime.invoke()` with a JWT bearer token and session ID

**`teardown.py` — Updated**
- Added `cleanup_runtime()` to delete the AgentCore Runtime and ECR repository

### Key Takeaways

- `BedrockAgentCoreApp` turns your agent into an HTTP service with `/invocations` and `/ping` endpoints — the same interface AWS uses for SageMaker and Lambda containers. Only 4 lines of code needed.
- JWT token propagation is the key security pattern: the same Cognito token that authenticates the user to the Runtime is forwarded to the Gateway for tool access. No re-authentication, no separate credentials.
- `session_id` from `context.session_id` provides short-term in-session context (conversation history). `actor_id` + AgentCore Memory provides long-term cross-session personalization. They work at different layers.
- `configure()` generates the Dockerfile — you never write it manually. The starter toolkit inspects your entrypoint and requirements.txt to build the right image.
- Deployment is asynchronous: `launch()` returns immediately, then you poll `status()`. This is the same pattern as KB ingestion and CloudFormation stack creation.
- Observability is automatic — AgentCore Runtime instruments your code with OpenTelemetry and sends traces to CloudWatch GenAI Observability with no extra code.

### Setup Order
```bash
python deploy_infrastructure.py   # 1. CloudFormation stack (if not already done)
python setup_gateway.py            # 2. Gateway + Lambda target (if not already done)
python create_memories.py          # 3. Memory store + seed data (if not already done)
python deploy_runtime.py           # 4. Deploy agent to AgentCore Runtime (5-10 min)
python invoke_runtime.py           # 5. Test the deployed agent
```

---

## Lab 5 — Online Evaluation with AgentCore Evaluations

### Original AWS Workshop Objectives
- Configure online evaluation to automatically assess agent performance in real-time
- Use built-in evaluators for goal success, correctness, and tool selection accuracy
- Generate test interactions and analyze quality metrics through CloudWatch dashboards

### How Online Evaluation Works

**What exactly are we evaluating?**

We're evaluating the quality of the agent's responses to customer interactions. Think of it as having a supervisor review every conversation and grading it:

```
Customer: "My laptop won't start up. Can you help?"
Agent:    [calls get_technical_support] → "Try these steps: 1. Check power cable..."

Evaluator 1 — GoalSuccessRate:
  "Did the agent actually help the customer with their problem?"
  → Score: 0.9 (yes, provided actionable troubleshooting steps)

Evaluator 2 — Correctness:
  "Is the troubleshooting advice factually accurate?"
  → Score: 0.85 (steps are correct, sourced from KB)

Evaluator 3 — ToolSelectionAccuracy:
  "Did the agent pick the right tool? (get_technical_support vs get_product_info)"
  → Score: 1.0 (correct — used get_technical_support for a troubleshooting question)
```

Without evaluation, the agent might hallucinate a return policy that doesn't exist (low Correctness), use `web_search` when it should use `get_technical_support` (low ToolSelectionAccuracy), or give a technically correct answer that doesn't solve the customer's problem (low GoalSuccessRate). Online evaluation catches these issues automatically before customers complain.

**How are the test cases generated?**

In this lab, the test cases in `run_eval_tests.py` are manually written prompts that cover different tool types. But these exist only to generate traffic for the demo — in production, you wouldn't need them. The online evaluation evaluates real customer conversations automatically:

```
Lab demo:     run_eval_tests.py → simulated traffic → evaluators score it
Production:   real customers    → real traffic       → evaluators score it automatically
```

Once the agent is live with real users, `run_eval_tests.py` becomes unnecessary. The evaluators just run on real conversations.

**Evaluation flow:**

```
Customer interacts with agent
         │
         ↓
AgentCore Runtime ──→ processes request ──→ returns response
         │
         │  (traces are captured automatically via OpenTelemetry)
         ↓
AgentCore Observability
         │
         │  (sampling: 100% for demo, 10-20% in production)
         ↓
Evaluators (LLM-based judges)
  ├── GoalSuccessRate:      Did the agent solve the customer's problem?
  ├── Correctness:          Is the information factually accurate?
  └── ToolSelectionAccuracy: Did the agent pick the right tool?
         │
         ↓
CloudWatch GenAI Observability Dashboard
  (scores, trends, low-scoring session investigation)
```

### What We Built

**`setup_evals.py` — Configure Online Evaluation**
- Retrieves the agent ARN from SSM (deployed in Lab 4)
- Creates an online evaluation config with 3 built-in evaluators at 100% sampling rate
- Evaluators are LLM-based — they use a model to judge the agent's responses asynchronously (no impact on response latency)

**`run_eval_tests.py` — Generate Test Interactions**
- Runs 5 diverse test scenarios against the production agent
- Each scenario exercises different tools and evaluator dimensions
- Results flow to CloudWatch for analysis

### Key Takeaways

- Online evaluation runs automatically on live traffic — no manual triggering needed. Once configured, every sampled session is scored.
- Evaluators are LLM-based judges that analyze completed session traces after the fact. They don't slow down the agent's response to the customer.
- The 3 built-in evaluators cover the most important quality dimensions: did the agent help (GoalSuccessRate), was it accurate (Correctness), and did it use the right tools (ToolSelectionAccuracy).
- Sampling rate controls cost vs coverage. 100% is great for demos but expensive in production. 10-20% gives good signal without evaluating every single session.
- Results appear in CloudWatch → GenAI Observability → Bedrock AgentCore. Look for per-session scores and aggregate trends over time.
- Low scores are actionable: low GoalSuccessRate → improve system prompt; low Correctness → update Knowledge Base; low ToolSelectionAccuracy → refine tool descriptions.

### Setup Order
```bash
python setup_evals.py       # 1. Configure online evaluation
python run_eval_tests.py    # 2. Generate test interactions
# Then check CloudWatch → GenAI Observability → Bedrock AgentCore
```

---

*More labs coming soon...*
