import boto3
import json

REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-sRFaSh6LsL"
CUSTOMER_ID = "CUST-MEMORY-001"
STRATEGY_ID = "UserPreferences-p8JsW9Hfv1"

client = boto3.client(
    "bedrock-agentcore",
    region_name=REGION
)

NAMESPACE = f"/strategies/{STRATEGY_ID}/actors/{CUSTOMER_ID}/"

print("=" * 70)
print("RETRIEVING MEMORY")
print("=" * 70)

print("Memory:", MEMORY_ID)
print("Customer:", CUSTOMER_ID)
print("Namespace:", NAMESPACE)

response = client.retrieve_memory_records(
    memoryId=MEMORY_ID,
    namespace=NAMESPACE,
    searchCriteria={
        "searchQuery": "What are the customer's preferred support language and explanation style?",
        "memoryStrategyId": STRATEGY_ID,
        "topK": 20,
    },
    maxResults=20,
)

print("\nRETRIEVE RESPONSE:")
print(json.dumps(response, indent=2, default=str))

records = response.get("memoryRecordSummaries", [])

print("\n" + "=" * 70)
print("RECORD COUNT:", len(records))
print("=" * 70)

for i, record in enumerate(records, 1):
    print(f"\nRECORD {i}:")
    print(json.dumps(record, indent=2, default=str))