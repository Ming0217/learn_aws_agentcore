#!/usr/bin/python
"""
lab_helpers/lab2_memory.py - AgentCore Memory Integration for Strands Agents
==============================================================================
This module provides:
  - ACTOR_ID / SESSION_ID constants for identifying the customer and session
  - CustomerSupportMemoryHooks: a Strands HookProvider that wires AgentCore Memory
    into the agent lifecycle automatically

STRANDS HOOKS EXPLAINED:
  Strands has a hook system that lets you intercept the agent's execution lifecycle
  at specific points without modifying the core agent logic. Think of it like
  middleware in a web framework.

  Two hook events are used here:
    - MessageAddedEvent: fires when a new message is added to the conversation.
      Used to retrieve relevant memories BEFORE the LLM processes the user's query.
    - AfterInvocationEvent: fires after the agent finishes responding.
      Used to save the interaction to memory AFTER the LLM has responded.

  This "before/after" pattern is what makes memory transparent — the agent code
  in main.py doesn't call any memory APIs directly. The hooks handle it all.

HOW MEMORY RETRIEVAL WORKS (retrieve_customer_context):
  1. User sends a message
  2. MessageAddedEvent fires
  3. Hook queries BOTH namespaces (semantic + preferences) using the user's message as the query
  4. Bedrock converts the query to an embedding and finds the most similar memory chunks
  5. Retrieved memories are prepended to the user's message as "Customer Context:"
  6. The LLM now sees both the context AND the query, enabling personalized responses

HOW MEMORY SAVING WORKS (save_support_interaction):
  1. Agent finishes responding
  2. AfterInvocationEvent fires
  3. Hook extracts the last user query + agent response from the message history
  4. Saves them as a new event via create_event()
  5. AgentCore asynchronously processes this event to update long-term memories
"""

import logging
import uuid

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType
from boto3.session import Session
from lab_helpers.utils import get_ssm_parameter, put_ssm_parameter
from strands.hooks import (
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

boto_session = Session()
REGION = boto_session.region_name

logger = logging.getLogger(__name__)

# ACTOR_ID: Unique identifier for the customer. Used to namespace all memory
# operations so this customer's memories are isolated from all others.
# In production, this would come from your auth system (e.g., Cognito user sub).
ACTOR_ID = "customer_001"

# SESSION_ID: Unique identifier for this conversation session.
# Generated fresh each time the module is loaded (i.e., each time main.py starts).
# Short-term memory is scoped to this ID; long-term memory is scoped to ACTOR_ID.
SESSION_ID = str(uuid.uuid4())

memory_client = MemoryClient(region_name=REGION)
memory_name = "CustomerSupportMemory"


def create_or_get_memory_resource():
    """
    Retrieves an existing memory store from SSM, or creates a new one if it doesn't exist.

    LEARNING NOTE: This is the "get or create" pattern — safe to call multiple times.
    It first checks SSM for an existing memory_id, then validates it's still active
    by calling get_memory(). If either check fails, it creates a fresh memory store.

    event_expiry_days=90 means raw conversation events (short-term memory) are
    automatically deleted after 90 days. Long-term memory records have separate TTLs.
    """
    try:
        memory_id = get_ssm_parameter("/app/customersupport/agentcore/memory_id")
        memory_client.gmcp_client.get_memory(memoryId=memory_id)
        return memory_id
    except Exception:
        try:
            strategies = [
                {
                    StrategyType.USER_PREFERENCE.value: {
                        "name": "CustomerPreferences",
                        "description": "Captures customer preferences and behavior",
                        "namespaces": ["support/customer/{actorId}/preferences"],
                    }
                },
                {
                    StrategyType.SEMANTIC.value: {
                        "name": "CustomerSupportSemantic",
                        "description": "Stores facts from conversations",
                        "namespaces": ["support/customer/{actorId}/semantic"],
                    }
                },
            ]
            print("Creating AgentCore Memory resources. This can take a couple of minutes...")
            # create_memory_and_wait blocks until the memory store is ACTIVE,
            # unlike create_memory which returns immediately while provisioning continues.
            response = memory_client.create_memory_and_wait(
                name=memory_name,
                description="Customer support agent memory",
                strategies=strategies,
                event_expiry_days=90,
            )
            memory_id = response["id"]
            put_ssm_parameter("/app/customersupport/agentcore/memory_id", memory_id)
            return memory_id
        except Exception:
            return None


def delete_memory(memory_hook):
    """
    Deletes the memory store and its SSM parameter.
    Used for cleanup/reset during development or lab teardown.
    """
    try:
        ssm_client = boto3.client("ssm", region_name=REGION)
        memory_client.delete_memory(memory_id=memory_hook.memory_id)
        ssm_client.delete_parameter(Name="/app/customersupport/agentcore/memory_id")
    except Exception:
        pass


class CustomerSupportMemoryHooks(HookProvider):
    """
    Strands HookProvider that integrates AgentCore Memory into the agent lifecycle.

    Implements two hooks:
      - retrieve_customer_context: injects relevant memories before LLM processing
      - save_support_interaction: persists the interaction after LLM responds

    LEARNING NOTE: By implementing HookProvider and registering via register_hooks(),
    this class plugs into Strands' event system cleanly. The agent in main.py just
    needs to accept this as a hook_provider — no memory logic leaks into agent code.
    """

    def __init__(self, memory_id: str, client: MemoryClient, actor_id: str, session_id: str):
        self.memory_id = memory_id
        self.client = client
        self.actor_id = actor_id
        self.session_id = session_id
        # Fetch the namespace paths for each strategy type from the memory store.
        # This avoids hardcoding namespace strings — they're defined in the memory config.
        self.namespaces = {
            i["type"]: i["namespaces"][0]
            for i in self.client.get_memory_strategies(self.memory_id)
        }

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """
        Hook: fires when a new message is added to the conversation.
        Retrieves relevant memories and prepends them to the user's message.

        LEARNING NOTE: We only retrieve on user messages (role == "user") and skip
        tool results (toolResult in content) to avoid injecting context into
        intermediate tool-calling steps where it's not needed.

        The retrieved context is injected directly into the message text as:
          "Customer Context:\n[PREFERENCE] ...\n[SEMANTIC] ...\n\n<original query>"
        This is a simple but effective way to give the LLM memory context without
        modifying the agent's system prompt or tool definitions.
        """
        messages = event.agent.messages
        if (
            messages[-1]["role"] == "user"
            and "toolResult" not in messages[-1]["content"][0]
        ):
            user_query = messages[-1]["content"][0]["text"]

            try:
                all_context = []

                for context_type, namespace in self.namespaces.items():
                    # Query each namespace separately — preferences and semantic
                    # memories are stored in different namespaces and retrieved independently.
                    # top_k=3 limits results per namespace to keep context concise.
                    memories = self.client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=namespace.format(actorId=self.actor_id),
                        query=user_query,
                        top_k=3,
                    )
                    for memory in memories:
                        if isinstance(memory, dict):
                            content = memory.get("content", {})
                            if isinstance(content, dict):
                                text = content.get("text", "").strip()
                                if text:
                                    # Tag each memory with its type for LLM clarity
                                    all_context.append(f"[{context_type.upper()}] {text}")

                if all_context:
                    context_text = "\n".join(all_context)
                    original_text = messages[-1]["content"][0]["text"]
                    messages[-1]["content"][0]["text"] = (
                        f"Customer Context:\n{context_text}\n\n{original_text}"
                    )
                    logger.info(f"Retrieved {len(all_context)} customer context items")

            except Exception as e:
                logger.error(f"Failed to retrieve customer context: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        """
        Hook: fires after the agent finishes its full response.
        Saves the last user query + agent response as a new memory event.

        LEARNING NOTE: We walk the message history in reverse to find the most
        recent user message and assistant response, skipping tool result messages.
        This ensures we only save the meaningful conversational exchange, not
        intermediate tool-calling steps.

        After saving, AgentCore asynchronously processes this event to update
        long-term memories — no further action needed here.
        """
        try:
            messages = event.agent.messages
            if len(messages) >= 2 and messages[-1]["role"] == "assistant":
                customer_query = None
                agent_response = None

                for msg in reversed(messages):
                    if msg["role"] == "assistant" and not agent_response:
                        agent_response = msg["content"][0]["text"]
                    elif (
                        msg["role"] == "user"
                        and not customer_query
                        and "toolResult" not in msg["content"][0]
                    ):
                        customer_query = msg["content"][0]["text"]
                        break

                if customer_query and agent_response:
                    self.client.create_event(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=self.session_id,
                        messages=[
                            (customer_query, "USER"),
                            (agent_response, "ASSISTANT"),
                        ],
                    )
                    logger.info("Saved support interaction to memory")

        except Exception as e:
            logger.error(f"Failed to save support interaction: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:
        """
        Registers both hooks with the Strands HookRegistry.

        LEARNING NOTE: register_hooks() is called automatically by Strands when
        the HookProvider is passed to the Agent. The registry maps event types
        to callback functions — Strands calls them at the right lifecycle points.
        """
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)
        logger.info("Customer support memory hooks registered")
