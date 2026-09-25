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
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client 
from botocore.config import Config

import argparse
import json
import os
import asyncio
import boto3
import logging
import uuid
from typing import Dict

from strands.hooks import (
    HookProvider,
    AfterInvocationEvent,
    HookRegistry,
    MessageAddedEvent,
)

from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")


# ── TODO 1 — App Initialisation ───────────────────────────────────────────────

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts in headless deployment.
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
#
# Gateway is in us-west-1.
# Knowledge Base, Memory and Code Interpreter are in us-east-1.

GATEWAY_URL = (
    "https://customersupportgateway-cok9isr3ra.gateway."
    "bedrock-agentcore.us-west-1.amazonaws.com/mcp"
)

KB_ID = "XSDMD9LKXO"

REGION = "us-east-1"

MEMORY_ID = "CustomerSupportMemory-sRFaSh6LsL"

GATEWAY_REGION = "us-west-1"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(
    model_id=model_id,
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
        retries={"max_attempts": 2},
    ),
)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """
    Return a dictionary mapping memory strategy type to namespace template.
    """

    try:
        response = mem_client.get_memory_strategies(
            memory_id=memory_id
        )

        strategies = response.get("strategies", [])

        namespaces = {}

        for strategy in strategies:
            # Strategy type can be represented slightly differently
            # depending on SDK/API version.
            strategy_type = (
                strategy.get("strategyType")
                or strategy.get("type")
                or strategy.get("strategy_type")
            )

            if not strategy_type:
                # Look inside the strategy object if needed.
                for key in strategy.keys():
                    upper_key = str(key).upper()

                    if "USER_PREFERENCE" in upper_key:
                        strategy_type = "USER_PREFERENCE"
                        break

                    if "SEMANTIC" in upper_key:
                        strategy_type = "SEMANTIC"
                        break

                    if "SUMMARY" in upper_key:
                        strategy_type = "SUMMARY"
                        break

                    if "EPISODIC" in upper_key:
                        strategy_type = "EPISODIC"
                        break

            namespace_templates = strategy.get("namespaceTemplates")

            if not namespace_templates:
                namespace_templates = strategy.get("namespaces")

            if namespace_templates:
                namespace = namespace_templates[0]

                if strategy_type:
                    namespaces[strategy_type.upper()] = namespace

        logger.info("Memory namespaces: %s", namespaces)

        return namespaces

    except Exception as e:
        logger.warning("Could not load memory strategies: %s", e)
        return {}


# ── Helper functions for memory ────────────────────────────────────────────────

def _extract_text_from_message(message) -> str:
    """
    Extract plain text from a Strands message.
    """

    if not isinstance(message, dict):
        return ""

    content = message.get("content", [])

    if isinstance(content, str):
        return content.strip()

    texts = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue

            text_value = block.get("text")

            if isinstance(text_value, str):
                texts.append(text_value)

    return "\n".join(texts).strip()


def _is_user_message(message) -> bool:
    if not isinstance(message, dict):
        return False

    return message.get("role") == "user"


def _is_assistant_message(message) -> bool:
    if not isinstance(message, dict):
        return False

    return message.get("role") == "assistant"


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

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

        self.namespaces = get_namespaces(
            memory_client,
            memory_id,
        )

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """
        Retrieve relevant memories and prepend them to the user message.
        """

        try:
            message = event.message

            if not _is_user_message(message):
                return

            query = _extract_text_from_message(message)

            if not query:
                return

            memories_found = []

            for strategy_type, namespace_template in self.namespaces.items():

                namespace = namespace_template.replace(
                    "{actorId}",
                    self.actor_id,
                )

                # Some namespace templates can use actorId without braces.
                namespace = namespace.replace(
                    "{actor_id}",
                    self.actor_id,
                )

                try:
                    response = self.memory_client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=namespace,
                        search_query=query,
                        top_k=5,
                    )

                    results = response.get("memoryRecords", [])

                    for record in results:

                        memory_text = (
                            record.get("content", {}).get("text")
                            if isinstance(record.get("content"), dict)
                            else record.get("content")
                        )

                        if memory_text:
                            memories_found.append(
                                f"[{strategy_type}] {memory_text}"
                            )

                except Exception as strategy_error:
                    logger.warning(
                        "Memory retrieval failed for %s: %s",
                        strategy_type,
                        strategy_error,
                    )

            if not memories_found:
                return

            context_text = (
                "Customer Context:\n"
                + "\n".join(memories_found)
                + "\n\n"
                + query
            )

            # Replace the current user message with the enriched message.
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if isinstance(block, dict) and "text" in block:
                        block["text"] = context_text
                        break

        except Exception as e:
            logger.warning(
                "Could not retrieve customer memory context: %s",
                e,
            )

    def save_support_interaction(self, event: AfterInvocationEvent):
        """
        Save the latest user query and assistant response to memory.
        """

        try:
            messages = event.agent.messages

            customer_query = None
            agent_response = None

            # Walk backwards through conversation history.
            for message in reversed(messages):

                if customer_query is None and _is_user_message(message):
                    text_value = _extract_text_from_message(message)

                    if text_value:
                        customer_query = text_value

                elif agent_response is None and _is_assistant_message(message):
                    text_value = _extract_text_from_message(message)

                    if text_value:
                        agent_response = text_value

                if customer_query and agent_response:
                    break

            if not customer_query or not agent_response:
                logger.warning(
                    "Could not find user/assistant messages to save."
                )
                return

            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )

            logger.info(
                "Saved support interaction for customer %s",
                self.actor_id,
            )

        except Exception as e:
            logger.warning(
                "Could not save support interaction: %s",
                e,
            )

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register both memory callbacks."""

        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.

    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Do not use this tool for live order tracking or refund operations.

    Args:
        query: The question or topic to search for.

    Returns:
        Relevant information retrieved from the knowledge base.
    """

    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                "text": query,
            },
        )

        results = response.get("retrievalResults", [])

        if not results:
            return "No relevant information was found in the knowledge base."

        chunks = []

        for result in results:
            content = result.get("content", {})

            if isinstance(content, dict):
                text_value = content.get("text")
            else:
                text_value = str(content)

            if text_value:
                chunks.append(text_value)

        if not chunks:
            return "No relevant information was found in the knowledge base."

        return "\n---\n".join(chunks)

    except Exception as e:
        logger.exception("Knowledge base search failed.")
        return f"Knowledge base search failed: {e}"


# ── TODO 7 — Loyalty Discount Tool ──────────────────────────────────────────

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter.

    The calculation includes points redemption, tier discount,
    points earned and remaining points.

    Args:
        loyalty_points:
            Customer's current points balance.

        tier:
            Customer tier — Silver, Gold, or Platinum.

        order_total:
            Order total in USD.

        product_category:
            standard, device, or fresh.

    Returns:
        Full discount breakdown and final price.
    """

    # Validate/sanitize inputs before injecting them into the
    # self-contained Python program.
    loyalty_points = max(0, int(loyalty_points))
    order_total = max(0.0, float(order_total))

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

    # Self-contained calculation code.
    code = f"""
import json
import math

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
max_redeemable_points = int(max_points_value / 0.01)

points_redeemed = min(
    (loyalty_points // 500) * 500,
    max_redeemable_points
)

# Keep redemption in 500-point increments.
points_redeemed = (points_redeemed // 500) * 500

points_discount = points_redeemed / 100.0

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
    * earn_rates.get(product_category, 1)
)

remaining_points = (
    loyalty_points - points_redeemed
    + points_earned
)

result = {{
    "order_total": round(order_total, 2),
    "tier": tier,
    "tier_discount_rate": tier_discount_rate,
    "tier_discount": round(tier_discount, 2),
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
    "product_category": product_category
}}

print(json.dumps(result))
"""

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

            for event in response.get("stream", []):

                if "result" in event:
                    return json.dumps(
                        event["result"],
                        default=str,
                    )

            return json.dumps(
                {
                    "error": "Code Interpreter returned no result."
                }
            )

    except Exception as e:

        logger.warning(
            "Code Interpreter unavailable: %s",
            e,
        )

        # Fallback calculation.
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
            order_total * discount_rate
        )

        final_total = max(
            0.0,
            order_total - tier_discount,
        )

        return json.dumps(
            {
                "order_total": round(order_total, 2),
                "tier": tier,
                "tier_discount_rate": discount_rate,
                "tier_discount": round(tier_discount, 2),
                "points_redeemed": 0,
                "points_discount": 0.0,
                "final_total": round(final_total, 2),
                "total_savings": round(tier_discount, 2),
                "points_earned": int(final_total),
                "remaining_points": loyalty_points,
                "fallback": True,
                "reason": str(e),
            }
        )


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:

      prompt:
        Customer's message.

      customer_id:
        Unique customer identifier.

      session_id:
        Session identifier.
    """

    try:

        user_input = payload.get("prompt", "")

        if not user_input:
            return "Please provide a customer support message."

        actor_id = payload.get(
            "customer_id",
            "anonymous-customer",
        )

        session_id = payload.get(
            "session_id"
        ) or str(uuid.uuid4())

        # Memory hook for cross-session customer context.
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # AgentCore Browser.
        agent_core_browser = AgentCoreBrowser(
            region=REGION
        )

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        # Gateway MCP client.
        mcp_client = MCPClient(
            lambda: streamable_http_client(
                GATEWAY_URL
            )
        )

        # MCP connection must remain open while the agent uses its tools.
        with mcp_client:

            gateway_tools = mcp_client.list_tools_sync()

            logger.info(
                "Loaded %d Gateway tools",
                len(gateway_tools),
            )

            tools.extend(gateway_tools)

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

- Use customer context from memory naturally when relevant.

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

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=system_prompt,
            )

            response = await agent.invoke_async(
                user_input
            )

            # Strands responses normally expose the final message.
            try:
                content = response.message.get(
                    "content",
                    []
                )

                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and isinstance(block.get("text"), str)
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


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    """Run one invocation from the command line for local testing."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "payload",
        type=str,
    )

    args = parser.parse_args()

    response = asyncio.run(
        invoke(
            json.loads(args.payload)
        )
    )

    print(response)


if __name__ == "__main__":
    app.run()
