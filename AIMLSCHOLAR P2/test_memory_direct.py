import boto3
import json
import time
from datetime import datetime, timezone

REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-sRFaSh6LsL"
CUSTOMER_ID = "CUST-MEMORY-001"
STRATEGY_ID = "UserPreferences-p8JsW9Hfv1"

client = boto3.client(
    "bedrock-agentcore",
    region_name=REGION,
)

NAMESPACE = f"/strategies/{STRATEGY_ID}/actors/{CUSTOMER_ID}/"

print("=" * 70)
print("INGESTING LONG-TERM MEMORY")
print("=" * 70)

print("Memory:", MEMORY_ID)
print("Customer:", CUSTOMER_ID)
print("Strategy:", STRATEGY_ID)
print("Namespace:", NAMESPACE)

response = client.ingest_data(
    memoryId=MEMORY_ID,
    actorId=CUSTOMER_ID,
    sessionId=f"memory-test-{int(time.time())}",
    contentTimestamp=datetime.now(timezone.utc),
    source={
        "inline": {
            "payload": [
                {
                    "conversational": {
                        "content": {
                            "text": (
                                "My preferred support language is Marathi. "
                                "I prefer concise explanations using bullet points."
                            )
                        },
                        "role": "USER",
                    }
                },
                {
                    "conversational": {
                        "content": {
                            "text": (
                                "Understood. I will provide customer support "
                                "in Marathi with concise bullet-point explanations."
                            )
                        },
                        "role": "ASSISTANT",
                    }
                },
            ]
        }
    },
)

print("\nINGEST RESPONSE:")
print(json.dumps(response, indent=2, default=str))

print("\nWaiting for extraction...")
time.sleep(60)

print("\n" + "=" * 70)
print("SEARCHING LONG-TERM MEMORY")
print("=" * 70)

response = client.retrieve_memory_records(
    memoryId=MEMORY_ID,
    namespace=NAMESPACE,
    searchCriteria={
        "searchQuery": "customer preferred support language and explanation style",
        "memoryStrategyId": STRATEGY_ID,
        "topK": 10,
    },
    maxResults=10,
)

print("\nRETRIEVE RESPONSE:")
print(json.dumps(response, indent=2, default=str))

records = response.get("memoryRecordSummaries", [])

print("\n" + "=" * 70)
print("FOUND:", len(records), "MEMORIES")
print("=" * 70)

for i, record in enumerate(records, 1):
    print(f"\nMEMORY {i}:")
    print(json.dumps(record, indent=2, default=str))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)