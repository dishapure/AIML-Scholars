import boto3
import json
import uuid
import time

client = boto3.client(
"bedrock-agentcore",
region_name="us-east-1"
)

agent_arn = (
"arn:aws:bedrock-agentcore:us-east-1:"
"041058731288:runtime/customer_support_agent-7h5E8b8lYM"
)

CUSTOMER_ID = "CUST-MEMORY-001"

def invoke(session_id, prompt):
        response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=session_id,
        payload=json.dumps({
        "prompt": prompt,
        "customer_id": CUSTOMER_ID
        }).encode(),
        qualifier="DEFAULT",
)


        body = response["response"].read().decode()

        print("=" * 70)
        print("CUSTOMER ID:", CUSTOMER_ID)
        print("SESSION ID:", session_id)
        print("RESPONSE:")
        print(body)
        print("=" * 70)

        return body


# ============================================================

# SESSION A — SAVE A DISTINCTIVE CUSTOMER PREFERENCE

# ============================================================

session_a = str(uuid.uuid4())

prompt_a = """
Remember this customer preference for future support conversations:

The customer's preferred support language is Marathi, and they
prefer concise explanations with bullet points.

Please confirm that you have recorded this preference.
"""

print("\nSESSION A — SAVING MEMORY\n")

invoke(session_a, prompt_a)

# Give AgentCore Memory a little time to process the event.

print("\nWaiting for memory processing...\n")
time.sleep(10)

# ============================================================

# SESSION B — NEW SESSION, SAME CUSTOMER

# ============================================================

session_b = str(uuid.uuid4())

prompt_b = """
What do you remember about this customer's support preferences?

Please specifically tell me:

1. Their preferred support language.
2. How they prefer explanations to be presented.

Do not ask me to provide the information again. Use the customer's
previously saved memory.
"""

print("\nSESSION B — RECALLING MEMORY\n")

invoke(session_b, prompt_b)

print("\nFINAL VERIFICATION")
print("Customer ID:", CUSTOMER_ID)
print("Session A:", session_a)
print("Session B:", session_b)
print("Different sessions:", session_a != session_b)
