"""
create_memories.py - AgentCore Memory Setup Script
====================================================
This is a ONE-TIME setup script that:
  1. Creates (or retrieves) the AgentCore Memory store with the right strategies
  2. Seeds it with historical customer interactions so the agent has context from day one
  3. Verifies that long-term memory processing completed successfully

RUN THIS BEFORE main.py. It saves the memory_id to SSM so main.py can reference it.

AGENTCORE MEMORY CONCEPTS:
  - Memory Store: The top-level container. One store per use case (e.g., CustomerSupportMemory).
  - Strategy: Defines HOW interactions are processed and stored. Two types used here:
      * UserPreference — LLM-powered extraction of behavioral patterns and preferences
        (e.g., "prefers ThinkPad", "budget under $1200", "uses Linux")
      * Semantic — LLM-powered extraction of factual statements from conversations
        (e.g., "owns MacBook Pro", "reported overheating issue", "order #MB-78432")
  - Namespace: A path that scopes memories per customer using {actorId} as a placeholder.
    This ensures customer A's memories are never mixed with customer B's.
  - Short-Term Memory: Raw conversation events (saved immediately via create_event).
    Scoped to a session_id. Used for within-session context.
  - Long-Term Memory: Processed summaries extracted from short-term events by an LLM.
    Scoped to actor_id. Persists across sessions. This is what enables personalization.

WHY SEED DATA?
  In a real system, long-term memories build up naturally over time as customers interact.
  For this lab, we seed historical interactions so we can immediately test memory recall
  without waiting for real conversations to accumulate.
"""

import logging
import time
from boto3.session import Session

from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType

from lab_helpers.utils import put_ssm_parameter

boto_session = Session()
REGION = boto_session.region_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 1: Create the Memory Store
# ---------------------------------------------------------------------------
# LEARNING NOTE: get_or_create_memory is idempotent — safe to run multiple times.
# If a memory store with this name already exists, it returns the existing one.
# This prevents duplicate stores from being created on re-runs.
# ---------------------------------------------------------------------------
memory_name = "CustomerSupportMemory"

memory_manager = MemoryManager(region_name=REGION)
memory = memory_manager.get_or_create_memory(
    name=memory_name,
    strategies=[
        {
            # UserPreference strategy: uses an LLM to infer customer preferences
            # from conversation history and store them as structured facts.
            # The {actorId} placeholder in the namespace is replaced at runtime
            # with the actual customer ID, isolating each customer's preferences.
            StrategyType.USER_PREFERENCE.value: {
                "name": "CustomerPreferences",
                "description": "Captures customer preferences and behavior",
                "namespaces": ["support/customer/{actorId}/preferences/"],
            }
        },
        {
            # Semantic strategy: uses an LLM to extract factual statements
            # from conversations (e.g., product owned, issue reported, order number).
            # These are stored as semantic memory chunks for retrieval later.
            StrategyType.SEMANTIC.value: {
                "name": "CustomerSupportSemantic",
                "description": "Stores facts from conversations",
                "namespaces": ["support/customer/{actorId}/semantic/"],
            }
        },
    ]
)

memory_id = memory["id"]

# Save memory_id to SSM so main.py can fetch it at runtime without hardcoding.
# LEARNING NOTE: This is the handoff point between setup and runtime.
# create_memories.py runs once → stores the ID → main.py reads it every time it starts.
put_ssm_parameter("/app/customersupport/agentcore/memory_id", memory_id)

if memory_id:
    print("✅ AgentCore Memory created successfully!")
    print(f"Memory ID: {memory_id}")
else:
    print("Memory resource not created. Try Again !")


# LEARNING NOTE: Import ACTOR_ID after memory creation to keep the import
# close to where it's used. ACTOR_ID is the customer identifier used to
# namespace all memory operations for this specific customer.
from lab_helpers.lab2_memory import ACTOR_ID


# ---------------------------------------------------------------------------
# STEP 2: Seed with historical customer interactions
# ---------------------------------------------------------------------------
# LEARNING NOTE: create_event() saves raw conversation turns as short-term memory.
# AgentCore then asynchronously processes these events in the background,
# running an LLM extraction pass to populate long-term memory (preferences + semantic).
# This processing is NOT instant — hence the polling loop below.
#
# Message format: list of (text, role) tuples where role is "USER" or "ASSISTANT".
# The session_id "previous_session" is a fixed string here since this is historical
# data, not a live session. In production, each real conversation gets a unique session_id.
# ---------------------------------------------------------------------------
previous_interactions = [
    ("I'm having issues with my MacBook Pro overheating during video editing.", "USER"),
    (
        "I can help with that thermal issue. For video editing workloads, let's check your Activity Monitor and adjust performance settings. Your MacBook Pro order #MB-78432 is still under warranty.",
        "ASSISTANT",
    ),
    (
        "What's the return policy on gaming headphones? I need low latency for competitive FPS games",
        "USER",
    ),
    (
        "For gaming headphones, you have 30 days to return. Since you're into competitive FPS, I'd recommend checking the audio latency specs - most gaming models have <40ms latency.",
        "ASSISTANT",
    ),
    (
        "I need a laptop under $1200 for programming. Prefer 16GB RAM minimum and good Linux compatibility. I like ThinkPad models.",
        "USER",
    ),
    (
        "Perfect! For development work, I'd suggest looking at our ThinkPad E series or Dell XPS models. Both have excellent Linux support and 16GB RAM options within your budget.",
        "ASSISTANT",
    ),
]

if memory_id:
    try:
        memory_client = MemoryClient(region_name=REGION)
        memory_client.create_event(
            memory_id=memory_id,
            actor_id=ACTOR_ID,
            session_id="previous_session",  # Fixed ID for seeded historical data
            messages=previous_interactions,
        )
        print("✅ Seeded customer history successfully")
        print("📝 Interactions saved to Short-Term Memory")
        print("⏳ Long-Term Memory processing will begin automatically...")
    except Exception as e:
        print(f"⚠️ Error seeding history: {e}")


# ---------------------------------------------------------------------------
# STEP 3: Poll until long-term memory processing completes
# ---------------------------------------------------------------------------
# LEARNING NOTE: Long-term memory extraction is asynchronous — AgentCore runs
# an LLM in the background to process the raw events into structured memories.
# We poll retrieve_memories() until results appear, with a timeout to avoid
# blocking forever if processing is slow or the service is under load.
#
# max_retries=6 with 10s sleep = ~1 minute total wait time.
# ---------------------------------------------------------------------------
print("🔍 Checking for processed Long-Term Memories...")
retries = 0
max_retries = 6

while retries < max_retries:
    memories = memory_client.retrieve_memories(
        memory_id=memory_id,
        namespace=f"support/customer/{ACTOR_ID}/preferences/",
        query="can you summarize the support issue",
    )

    if memories:
        print(f"✅ Found {len(memories)} preference memories after {retries * 10} seconds!")
        break

    retries += 1
    if retries < max_retries:
        print(f"⏳ Still processing... waiting 10 more seconds (attempt {retries}/{max_retries})")
        time.sleep(10)
    else:
        print("⚠️ Memory processing is taking longer than expected. This can happen with overloading..")
        break

# Display extracted preference memories
# LEARNING NOTE: Each memory item is a dict with a nested content.text field.
# The LLM has already summarized and structured the raw conversation into
# discrete preference statements — notice how it infers intent, not just quotes text.
print("🎯 AgentCore Memory automatically extracted these customer preferences from our seeded conversations:")
print("=" * 80)
for i, memory in enumerate(memories, 1):
    if isinstance(memory, dict):
        content = memory.get("content", {})
        if isinstance(content, dict):
            text = content.get("text", "")
            print(f"  {i}. {text}")


# ---------------------------------------------------------------------------
# STEP 4: Verify semantic memory extraction
# ---------------------------------------------------------------------------
# LEARNING NOTE: We poll semantic memories separately because they're stored
# in a different namespace. This confirms both strategy types are working.
# The while True loop here assumes processing will eventually complete —
# in production you'd add a timeout/retry limit here too.
# ---------------------------------------------------------------------------
while True:
    semantic_memories = memory_client.retrieve_memories(
        memory_id=memory_id,
        namespace=f"support/customer/{ACTOR_ID}/semantic/",
        query="information on the technical support issue",
    )
    print("🧠 AgentCore Memory identified these factual details from conversations:")
    print("=" * 80)
    if semantic_memories:
        break
    time.sleep(10)

for i, memory in enumerate(semantic_memories, 1):
    if isinstance(memory, dict):
        content = memory.get("content", {})
        if isinstance(content, dict):
            text = content.get("text", "")
            print(f"  {i}. {text}")

