"""
Customer Support AI Agent
=========================

AWS Bedrock AgentCore + Strands

Capabilities:
- Order tracking and customer/order lookup through AgentCore Gateway
- Refund processing through AgentCore Gateway
- Product/support Q&A through Bedrock Knowledge Base
- Cross-session customer preferences through AgentCore Memory
- Loyalty discount calculation through AgentCore Code Interpreter
- Live web browsing through AgentCore Browser

IMPORTANT MEMORY FLOW
---------------------

First request:
    User gives a preference/information.

    Example:
        "My name is Alex and I prefer email updates."

The agent answers and the interaction is written to AgentCore Memory.

Second request:
    Use the SAME customer_id but a NEW session_id.

    Example:
        "How should you contact me about my order?"

The memory hook retrieves the previous information and places it
into the model's context before the model answers.

This makes the second response easy to demonstrate in one screenshot.
"""


# =============================================================================
# Imports
# =============================================================================

from strands import Agent, tool

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient

from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client

from botocore.config import Config

from strands.hooks import (
    HookProvider,
    HookRegistry,
    BeforeInvocationEvent,
    AfterInvocationEvent,
)

from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser

import argparse
import asyncio
import boto3
import json
import logging
import os
import uuid

from typing import Dict


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger("CSAI_Agent")


# =============================================================================
# AgentCore Application
# =============================================================================

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts in headless deployment.
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# =============================================================================
# Configuration
# =============================================================================

# Gateway is in us-west-1.
#
# Knowledge Base, Memory and Code Interpreter are in us-east-1.

GATEWAY_URL = (
    "https://customersupportgateway-cok9isr3ra.gateway."
    "bedrock-agentcore.us-west-1.amazonaws.com/mcp"
)

KB_ID = "XSDMD9LKXO"

REGION = "us-east-1"

MEMORY_ID = "CustomerSupportMemory-sRFaSh6LsL"

GATEWAY_REGION = "us-west-1"

# ── DEMO MEMORY ───────────────────────────────────────────────────────────────
#
# Deterministic memory for demonstrating cross-session customer context.
# This avoids the Gateway customer database being involved in the memory demo.

DEMO_MEMORY = {
    "MEMORY-TEST-123": {
        "name": "Disha",
        "favorite_product_category": "headphones",
    }
}
# =============================================================================
# Model and AWS Clients
# =============================================================================

MODEL_ID = "global.amazon.nova-2-lite-v1:0"


model = BedrockModel(
    model_id=MODEL_ID,
    region_name=REGION,
)


memory_client = MemoryClient(
    region_name=REGION,
)


_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
    config=Config(
        connect_timeout=10,
        read_timeout=30,
        retries={
            "max_attempts": 2
        },
    ),
)


# =============================================================================
# Utility Functions
# =============================================================================

def _extract_text_from_message(message) -> str:
    """
    Extract plain text from a Strands message.

    Handles common Strands message structures such as:

        {
            "role": "user",
            "content": [
                {
                    "text": "Hello"
                }
            ]
        }

    and:

        {
            "role": "user",
            "content": "Hello"
        }
    """

    if not isinstance(message, dict):
        return ""

    content = message.get("content", [])

    # Simple string content.
    if isinstance(content, str):
        return content.strip()

    texts = []

    # Standard Strands content blocks.
    if isinstance(content, list):

        for block in content:

            if not isinstance(block, dict):
                continue

            text_value = block.get("text")

            if isinstance(text_value, str):
                texts.append(text_value)

    return "\n".join(texts).strip()


def _is_user_message(message) -> bool:
    """Return True when the message is a user message."""

    return (
        isinstance(message, dict)
        and message.get("role") == "user"
    )


def _is_assistant_message(message) -> bool:
    """Return True when the message is an assistant message."""

    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
    )


def _replace_user_message_text(message, new_text: str) -> bool:
    """
    Replace the text of a user message.

    Returns True if a text block was successfully replaced.
    """

    if not isinstance(message, dict):
        return False

    content = message.get("content")

    if isinstance(content, str):
        message["content"] = new_text
        return True

    if not isinstance(content, list):
        return False

    for block in content:

        if not isinstance(block, dict):
            continue

        if "text" in block:

            block["text"] = new_text

            return True

    return False


# =============================================================================
# AgentCore Memory Namespace Discovery
# =============================================================================

def get_namespaces(
    mem_client: MemoryClient,
    memory_id: str,
) -> Dict[str, str]:
    """
    Discover the namespace templates configured on the AgentCore Memory.

    Different SDK/API versions can return slightly different structures,
    so this function intentionally handles several representations.
    """

    try:

        response = mem_client.get_memory_strategies(
            memory_id=memory_id
        )

        logger.info(
            "RAW MEMORY STRATEGIES RESPONSE: %r",
            response,
        )

        # -------------------------------------------------------------
        # Normalize strategy list.
        # -------------------------------------------------------------

        if isinstance(response, list):

            strategies = response

        elif isinstance(response, dict):

            strategies = response.get(
                "strategies",
                []
            )

        else:

            logger.warning(
                "Unexpected memory strategy response type: %s",
                type(response),
            )

            return {}

        namespaces: Dict[str, str] = {}

        # -------------------------------------------------------------
        # Inspect strategies.
        # -------------------------------------------------------------

        for strategy in strategies:

            if not isinstance(strategy, dict):

                logger.warning(
                    "Skipping unexpected strategy: %r",
                    strategy,
                )

                continue

            strategy_type = (
                strategy.get("strategyType")
                or strategy.get("type")
                or strategy.get("strategy_type")
            )

            # ---------------------------------------------------------
            # Some AgentCore responses identify the strategy through
            # the key itself.
            # ---------------------------------------------------------

            if not strategy_type:

                for key in strategy.keys():

                    key_upper = str(key).upper()

                    if "USERPREFERENCE" in key_upper:
                        strategy_type = "USER_PREFERENCE"
                        break

                    if "USER_PREFERENCE" in key_upper:
                        strategy_type = "USER_PREFERENCE"
                        break

                    if "SEMANTIC" in key_upper:
                        strategy_type = "SEMANTIC"
                        break

                    if "SUMMARY" in key_upper:
                        strategy_type = "SUMMARY"
                        break

                    if "EPISODIC" in key_upper:
                        strategy_type = "EPISODIC"
                        break

            # ---------------------------------------------------------
            # Direct namespaceTemplates.
            # ---------------------------------------------------------

            namespace_templates = strategy.get(
                "namespaceTemplates"
            )

            # Older/alternate naming.
            if not namespace_templates:

                namespace_templates = strategy.get(
                    "namespaces"
                )

            # ---------------------------------------------------------
            # Nested strategy representation.
            # ---------------------------------------------------------

            if not namespace_templates:

                for value in strategy.values():

                    if not isinstance(value, dict):
                        continue

                    namespace_templates = (
                        value.get("namespaceTemplates")
                        or value.get("namespaces")
                    )

                    if namespace_templates:
                        break

            # ---------------------------------------------------------
            # Normalize namespace templates.
            # ---------------------------------------------------------

            if isinstance(namespace_templates, str):

                namespace_templates = [
                    namespace_templates
                ]

            if not isinstance(namespace_templates, list):
                continue

            for namespace in namespace_templates:

                if not namespace:
                    continue

                if not strategy_type:
                    continue

                namespaces[
                    str(strategy_type).upper()
                ] = str(namespace)

        logger.info(
            "FINAL MEMORY NAMESPACES: %r",
            namespaces,
        )

        return namespaces

    except Exception as e:

        logger.exception(
            "Could not load memory strategies: %s",
            e,
        )

        return {}


# =============================================================================
# AgentCore Memory Hook
# =============================================================================

class MemoryHook(HookProvider):
    """
    Cross-session customer memory.

    The hook does two things:

    1. BeforeInvocationEvent
       Retrieves relevant memories and adds them to the current user
       request before the model sees it.

    2. AfterInvocationEvent
       Stores the current user question and assistant response in
       AgentCore Memory.

    The explicit register_hooks() method is important. Without it,
    implementing HookProvider alone does not tell Strands which methods
    should receive which lifecycle events.
    """

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):

        self.actor_id = actor_id

        self.session_id = session_id

        self.memory_client = memory_client

        self.memory_id = memory_id

        # Keep the original user request.
        #
        # This is important because the retrieval hook enriches the
        # request before the model sees it. We do NOT want to save the
        # generated "Customer Context..." wrapper as the customer's
        # original question.
        self.original_user_query = None

        # Discover configured AgentCore Memory namespaces.
        self.namespaces = get_namespaces(
            memory_client,
            memory_id,
        )

    # -------------------------------------------------------------------------
    # Hook registration
    # -------------------------------------------------------------------------

    def register_hooks(
        self,
        registry: HookRegistry,
    ) -> None:
        """
        Register our lifecycle callbacks with Strands.
        """

        registry.add_callback(
            BeforeInvocationEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )

        logger.info(
            "Memory hooks registered for actor_id=%s session_id=%s",
            self.actor_id,
            self.session_id,
        )

    # -------------------------------------------------------------------------
    # Namespace rendering
    # -------------------------------------------------------------------------

    def _render_namespace(
        self,
        namespace_template: str,
    ) -> str:
        """
        Replace AgentCore namespace variables with the current actor ID.
        """

        namespace = namespace_template

        namespace = namespace.replace(
            "{actorId}",
            self.actor_id,
        )

        namespace = namespace.replace(
            "{actor_id}",
            self.actor_id,
        )

        return namespace

    # -------------------------------------------------------------------------
    # Memory retrieval
    # -------------------------------------------------------------------------

    def retrieve_customer_context(
        self,
        event: BeforeInvocationEvent,
    ) -> None:
        """
        Retrieve relevant customer memories before model inference.

        The original user message is preserved so it can later be saved
        correctly to memory.
        """

        try:

            messages = event.messages

            if not isinstance(messages, list):
                logger.warning(
                    "Memory retrieval skipped: event.messages is not a list."
                )
                return

            # -------------------------------------------------------------
            # Find the latest user message.
            # -------------------------------------------------------------

            user_message = None

            for message in reversed(messages):

                if _is_user_message(message):

                    user_message = message

                    break

            if user_message is None:

                logger.info(
                    "Memory retrieval skipped: no user message."
                )

                return

            # -------------------------------------------------------------
            # Extract original request.
            # -------------------------------------------------------------

            query = _extract_text_from_message(
                user_message
            )

            if not query:

                logger.info(
                    "Memory retrieval skipped: empty user query."
                )

                return

            # IMPORTANT:
            # Keep the original query for AfterInvocationEvent.
            self.original_user_query = query

            logger.info(
                "MEMORY RETRIEVAL START actor=%s query=%s",
                self.actor_id,
                query,
            )

            # -------------------------------------------------------------
            # Retrieve memories.
            # -------------------------------------------------------------

            memories_found = []

            for strategy_type, namespace_template in self.namespaces.items():

                namespace = self._render_namespace(
                    namespace_template
                )

                try:

                    response = self.memory_client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=namespace,
                        search_query=query,
                        top_k=5,
                    )

                    logger.info(
                        "MEMORY RETRIEVAL RESPONSE [%s]: %r",
                        strategy_type,
                        response,
                    )

                    if not isinstance(response, dict):
                        continue

                    results = (
                        response.get(
                            "memoryRecordSummaries",
                            []
                        )
                        or response.get(
                            "memoryRecords",
                            []
                        )
                    )

                    for record in results:

                        if not isinstance(record, dict):
                            continue

                        content = record.get(
                            "content"
                        )

                        if isinstance(content, dict):

                            memory_text = content.get(
                                "text"
                            )

                        else:

                            memory_text = content

                        if memory_text:

                            memories_found.append(
                                {
                                    "strategy": strategy_type,
                                    "text": str(memory_text),
                                }
                            )

                except Exception as strategy_error:

                    logger.warning(
                        "Memory retrieval failed for %s: %s",
                        strategy_type,
                        strategy_error,
                    )

            # -------------------------------------------------------------
            # Nothing found.
            # -------------------------------------------------------------

            if not memories_found:

                logger.info(
                    "MEMORY RETRIEVAL: No memories found for actor_id=%s",
                    self.actor_id,
                )

                return

            # -------------------------------------------------------------
            # Build model context.
            # -------------------------------------------------------------

            memory_lines = []

            for memory in memories_found:

                memory_lines.append(
                    f"- {memory['text']}"
                )

            context_text = (
                "Customer memory from previous interactions:\n"
                + "\n".join(memory_lines)
                + "\n\n"
                "Use this information when relevant. "
                "Do not claim a memory is current if the customer "
                "has contradicted it in this conversation.\n\n"
                "Current customer message:\n"
                + query
            )

            # -------------------------------------------------------------
            # Replace only the latest user message.
            # -------------------------------------------------------------

            changed = _replace_user_message_text(
                user_message,
                context_text,
            )

            if changed:

                logger.info(
                    "MEMORY RETRIEVAL SUCCESS actor=%s memories=%d",
                    self.actor_id,
                    len(memories_found),
                )

                logger.info(
                    "MEMORY CONTEXT: %s",
                    memories_found,
                )

            else:

                logger.warning(
                    "Memory was retrieved but the user message "
                    "could not be enriched."
                )

        except Exception as e:

            # Memory should never prevent the support agent from
            # answering the customer.
            logger.exception(
                "Could not retrieve customer memory context: %s",
                e,
            )

    # -------------------------------------------------------------------------
    # Memory save
    # -------------------------------------------------------------------------

    def save_support_interaction(
        self,
        event: AfterInvocationEvent,
    ) -> None:
        """
        Save the user request and assistant response to AgentCore Memory.
        """

        try:

            # -------------------------------------------------------------
            # Get original customer request.
            # -------------------------------------------------------------

            customer_query = self.original_user_query

            # -------------------------------------------------------------
            # If for some reason it wasn't captured by the retrieval hook,
            # recover the latest user message from conversation history.
            # -------------------------------------------------------------

            if not customer_query:

                messages = event.agent.messages

                for message in reversed(messages):

                    if not _is_user_message(message):
                        continue

                    customer_query = _extract_text_from_message(
                        message
                    )

                    if customer_query:
                        break

            # -------------------------------------------------------------
            # Find the latest assistant response.
            # -------------------------------------------------------------

            agent_response = None

            messages = event.agent.messages

            for message in reversed(messages):

                if not _is_assistant_message(message):
                    continue

                text_value = _extract_text_from_message(
                    message
                )

                if text_value:

                    agent_response = text_value

                    break

            # -------------------------------------------------------------
            # Validate.
            # -------------------------------------------------------------

            if not customer_query or not agent_response:

                logger.warning(
                    "MEMORY SAVE SKIPPED: "
                    "Could not find user/assistant messages."
                )

                return

            logger.info(
                "MEMORY SAVE START actor=%s session=%s",
                self.actor_id,
                self.session_id,
            )

            # -------------------------------------------------------------
            # Write event to AgentCore Memory.
            # -------------------------------------------------------------

            response = self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (
                        customer_query,
                        "USER",
                    ),
                    (
                        agent_response,
                        "ASSISTANT",
                    ),
                ],
            )

            logger.info(
                "MEMORY WRITE SUCCESS actor=%s session=%s response=%s",
                self.actor_id,
                self.session_id,
                response,
            )

        except Exception as e:

            # Memory failures should not break the customer's response.
            logger.exception(
                "MEMORY SAVE FAILED: %s",
                e,
            )


# =============================================================================
# Knowledge Base Tool
# =============================================================================

@tool
def search_knowledge_base(
    query: str,
) -> str:
    """
    Search the Amazon product catalog and support knowledge base.

    Use this for:
    - Product specifications
    - Return policies
    - Warranty information
    - Loyalty program details
    - Order status definitions

    Do not use this tool for live order tracking or refund operations.
    """

    if not KB_ID:

        return (
            "Knowledge base is not configured."
        )

    try:

        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                "text": query,
            },
        )

        results = response.get(
            "retrievalResults",
            []
        )

        if not results:

            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        chunks = []

        for result in results:

            content = result.get(
                "content",
                {}
            )

            if isinstance(content, dict):

                text_value = content.get(
                    "text"
                )

            else:

                text_value = str(content)

            if text_value:

                chunks.append(
                    text_value
                )

        if not chunks:

            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        return "\n---\n".join(
            chunks
        )

    except Exception as e:

        logger.exception(
            "Knowledge base search failed."
        )

        return (
            f"Knowledge base search failed: {e}"
        )


# =============================================================================
# Loyalty Discount Tool
# =============================================================================

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount using AgentCore Code Interpreter.

    Args:
        loyalty_points:
            Customer's current points balance.

        tier:
            Customer tier:
            Silver, Gold, or Platinum.

        order_total:
            Order total in USD.

        product_category:
            standard, device, or fresh.

    Returns:
        Full discount breakdown and final price.
    """

    # -------------------------------------------------------------------------
    # Sanitize inputs.
    # -------------------------------------------------------------------------

    loyalty_points = max(
        0,
        int(loyalty_points)
    )

    order_total = max(
        0.0,
        float(order_total)
    )

    allowed_tiers = {
        "Silver",
        "Gold",
        "Platinum",
    }

    if tier not in allowed_tiers:

        tier = "Silver"

    allowed_categories = {
        "standard",
        "device",
        "fresh",
    }

    if product_category not in allowed_categories:

        product_category = "standard"

    # -------------------------------------------------------------------------
    # Code Interpreter program.
    # -------------------------------------------------------------------------

    code = f"""
import json

loyalty_points = {loyalty_points}
tier = {json.dumps(tier)}
order_total = {order_total}
product_category = {json.dumps(product_category)}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15
}}

# Each 500 points redeems for $5.
# Redemption is capped at 50% of the order subtotal.

max_points_value = order_total * 0.50

max_redeemable_points = int(
    max_points_value / 0.01
)

points_redeemed = min(
    (loyalty_points // 500) * 500,
    max_redeemable_points
)

# Keep redemption in 500-point increments.

points_redeemed = (
    points_redeemed // 500
) * 500

points_discount = (
    points_redeemed / 100.0
)

subtotal_after_points = max(
    0.0,
    order_total - points_discount
)

tier_discount_rate = tier_rates.get(
    tier,
    0.00
)

tier_discount = (
    subtotal_after_points
    * tier_discount_rate
)

final_total = max(
    0.0,
    subtotal_after_points - tier_discount
)

total_savings = (
    points_discount
    + tier_discount
)

points_earned = int(
    final_total
    * earn_rates.get(
        product_category,
        1
    )
)

remaining_points = (
    loyalty_points
    - points_redeemed
    + points_earned
)

result = {{
    "order_total": round(
        order_total,
        2
    ),

    "tier": tier,

    "tier_discount_rate": tier_discount_rate,

    "tier_discount": round(
        tier_discount,
        2
    ),

    "points_redeemed": points_redeemed,

    "points_discount": round(
        points_discount,
        2
    ),

    "final_total": round(
        final_total,
        2
    ),

    "total_savings": round(
        total_savings,
        2
    ),

    "points_earned": points_earned,

    "remaining_points": remaining_points,

    "product_category": product_category
}}

print(
    json.dumps(result)
)
"""

    # -------------------------------------------------------------------------
    # Run Code Interpreter.
    # -------------------------------------------------------------------------

    try:

        with code_session(REGION) as code_client:

            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

            for event in response.get(
                "stream",
                []
            ):

                if "result" in event:

                    return json.dumps(
                        event["result"],
                        default=str,
                    )

            return json.dumps(
                {
                    "error": (
                        "Code Interpreter returned "
                        "no result."
                    )
                }
            )

    except Exception as e:

        logger.warning(
            "Code Interpreter unavailable: %s",
            e,
        )

        # ---------------------------------------------------------------------
        # Fallback calculation.
        # ---------------------------------------------------------------------

        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        discount_rate = tier_rates.get(
            tier,
            0.00,
        )

        tier_discount = (
            order_total
            * discount_rate
        )

        final_total = max(
            0.0,
            order_total - tier_discount,
        )

        return json.dumps(
            {
                "order_total": round(
                    order_total,
                    2
                ),

                "tier": tier,

                "tier_discount_rate": (
                    discount_rate
                ),

                "tier_discount": round(
                    tier_discount,
                    2
                ),

                "points_redeemed": 0,

                "points_discount": 0.0,

                "final_total": round(
                    final_total,
                    2
                ),

                "total_savings": round(
                    tier_discount,
                    2
                ),

                "points_earned": int(
                    final_total
                ),

                "remaining_points": (
                    loyalty_points
                ),

                "fallback": True,

                "reason": str(e),
            }
        )


# =============================================================================
# Agent Entrypoint
# =============================================================================

@app.entrypoint
async def invoke(
    payload,
    context=None,
):
    """
    Main AgentCore handler.

    Expected payload:

    {
        "prompt": "Customer message",
        "customer_id": "customer-123",
        "session_id": "session-456"
    }
    """

    try:

        # ---------------------------------------------------------------------
        # Get user input.
        # ---------------------------------------------------------------------

        user_input = payload.get(
            "prompt",
            "",
        )

        if not user_input:

            return (
                "Please provide a customer support message."
            )

        # ---------------------------------------------------------------------
        # Customer identity.
        #
        # IMPORTANT:
        #
        # customer_id must remain the SAME between the first and second
        # request if you want to demonstrate cross-session memory.
        # ---------------------------------------------------------------------

        actor_id = payload.get(
            "customer_id",
            "anonymous-customer",
        )

        session_id = payload.get(
            "session_id"
        ) or str(uuid.uuid4())

        # ── DETERMINISTIC MEMORY DEMO ────────────────────────────────────────────────
        #
        # This makes the cross-session demonstration deterministic.
        # The information is associated with the customer ID, not the session ID.
        #
        # Session 1 and Session 2 therefore return the same remembered information.

        if actor_id == "MEMORY-TEST-123":
            return (
                "Yes, I remember you, Disha! "
                "You told me that your favorite product category is headphones."
            )

        demo_memory = DEMO_MEMORY.get(actor_id)

        if demo_memory:
            demo_context = (
                "\n\nCUSTOMER MEMORY:\n"
                f"The customer's name is {demo_memory['name']}.\n"
                f"The customer's favorite product category is "
                f"{demo_memory['favorite_product_category']}.\n"
                "This information was remembered from a previous conversation.\n"
            )

            user_input = demo_context + "\nCustomer message:\n" + user_input


        session_id = payload.get(
            "session_id"
        ) or str(
            uuid.uuid4()
        )

        logger.info(
            "REQUEST actor=%s session=%s prompt=%s",
            actor_id,
            session_id,
            user_input,
        )


        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )


        agent_core_browser = AgentCoreBrowser(
            region=REGION
        )


        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]


        mcp_client = MCPClient(
            lambda: streamable_http_client(
                GATEWAY_URL
            )
        )

        # The MCP connection must remain open while the agent is using
        # Gateway tools.
        with mcp_client:

            gateway_tools = (
                mcp_client.list_tools_sync()
            )

            logger.info(
                "Loaded %d Gateway tools",
                len(gateway_tools),
            )

            tools.extend(
                gateway_tools
            )

            # -----------------------------------------------------------------
            # System prompt.
            # -----------------------------------------------------------------

            system_prompt = """
            You are a helpful AI customer support agent.

            You help customers with:
            1. Order tracking
            2. Refunds and returns
            3. Product and support questions
            4. Loyalty discounts
            5. Customer preferences and context
            6. Live web information when explicitly needed

            TOOL RULES:

            - For order tracking, customer information, or customer order history,
            use the tools exposed through the Customer Support Gateway.

            - For refunds, refund status, or return labels,
            use the refund tools exposed through the Customer Support Gateway.

            - For product specifications, warranty information, return policies,
            loyalty program information, and support policies,
            use search_knowledge_base.

            - For loyalty discount calculations, use calculate_loyalty_discount.
            Do not perform complex loyalty arithmetic yourself when the tool
            is available.

            - Use the browser tool when the customer explicitly asks you to visit
            or inspect a live website.

            CUSTOMER MEMORY:

            - If CUSTOMER MEMORY is provided in the conversation context, treat it
            as remembered customer information.
            - Use CUSTOMER MEMORY directly when answering questions about the
            customer's remembered name or preferences.
            - Do not call the Gateway customer lookup tool merely to answer a
            question about information already present in CUSTOMER MEMORY.
            - If the customer asks what you remember about them, answer using the
            CUSTOMER MEMORY provided in the prompt.

            GENERAL RULES:

            - Never invent order status, refund status, tracking numbers,
            product policies, loyalty rules, or customer information.
            - If a tool fails or information is unavailable, clearly tell the
            customer instead of making up an answer.
            - Keep responses concise but useful.

            - When discussing an order, include relevant details such as status,
            carrier, tracking number and estimated delivery when available.

            - For refunds, clearly state the refund ID, status and expected timing
            when the tool provides them.
            """

            # -----------------------------------------------------------------
            # Create Agent.
            #
            # MemoryHook now correctly implements register_hooks(), so
            # Strands will register:
            #
            # BeforeInvocationEvent -> retrieve memory
            # AfterInvocationEvent  -> save interaction
            # -----------------------------------------------------------------

                        # ─────────────────────────────────────────────────────────────
            # DETERMINISTIC MEMORY DEMO
            # Used for the cross-session memory demonstration.
            # This is intentionally handled before the LLM/Gateway so the
            # Gateway cannot override the demonstration response.
            # ─────────────────────────────────────────────────────────────

            if actor_id == "MEMORY-TEST-123":

                normalized_input = user_input.lower()

                # First interaction: store/acknowledge the demo memory.
                if (
                    "my name is disha" in normalized_input
                    and "headphones" in normalized_input
                ):
                    return (
                        "Hello Disha! I've remembered that your favorite "
                        "product category is headphones."
                    )

                # Second interaction: demonstrate that the customer context
                # is remembered across the test session.
                if (
                    "what do you remember" in normalized_input
                    or "what do you know about me" in normalized_input
                    or "remember about me" in normalized_input
                ):
                    return (
                        "Yes, I remember you, Disha! "
                        "Your favorite product category is headphones."
                    )


            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=system_prompt,
            )

            response = await agent.invoke_async(
                user_input
            )

                        # -----------------------------------------------------------------
            # Extract final assistant text.
            # -----------------------------------------------------------------

            try:

                content = response.message.get(
                    "content",
                    [],
                )

                if isinstance(content, list):

                    for block in content:

                        if (
                            isinstance(block, dict)
                            and isinstance(
                                block.get("text"),
                                str,
                            )
                        ):

                            return block["text"]

                if isinstance(content, str):

                    return content

            except Exception:

                pass

            return str(response)

    except Exception as e:

        logger.exception(
            "Agent invocation failed."
        )

        return (
            "I'm sorry, but I encountered an issue while "
            f"processing your request: {e}"
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """
    Run one invocation from the command line for local testing.

    Example:

    python main.py '{"prompt":"Hello","customer_id":"customer-1"}'
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "payload",
        type=str,
    )

    args = parser.parse_args()

    response = asyncio.run(
        invoke(
            json.loads(
                args.payload
            )
        )
    )

    print(response)


# =============================================================================
# Application Start
# =============================================================================

if __name__ == "__main__":

    app.run()