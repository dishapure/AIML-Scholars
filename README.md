# 🤖 AI Customer Support Agent with Amazon Bedrock AgentCore

<p align="center">

**An AI-powered customer support agent with knowledge retrieval, browser interaction, and persistent customer memory.**

Built with **Amazon Bedrock · AgentCore · Python · boto3 · uv**

<br>

<a href="https://github.com/dishapure/AIML-Scholars"> 
<img src="https://img.shields.io/badge/GitHub-AIML--Scholars-181717?style=for-the-badge&logo=github" alt="GitHub">
</a>
<a href="https://aws.amazon.com/bedrock/">
<img src="https://img.shields.io/badge/Amazon%20Bedrock-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" alt="Amazon Bedrock">
</a>
<a href="https://aws.amazon.com/">
<img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS">
</a>
<a href="https://www.python.org/">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</a>

</p>

---

## ✨ Overview

This project is an **AI customer support agent** built using **Amazon Bedrock and Amazon Bedrock AgentCore**.

The system combines an AI agent with customer-support knowledge retrieval, browser-based interaction, and AgentCore Memory to create a support workflow that can retain useful customer preferences across conversations.

A major focus of the project is demonstrating the difference between:

* 💬 **Conversation/session events** — customer interactions stored against a specific actor and session.
* 🧠 **Long-term memory** — persistent customer preferences extracted and stored as structured memory records.

The completed memory demonstration successfully retrieves preferences such as:

> **Preferred support language:** Marathi

and

> **Preferred response style:** Concise explanations using bullet points.

---

# 🚀 What This Project Demonstrates

### 🤖 AI Customer Support...

The agent processes customer conversations and generates support responses using Amazon Bedrock.

### 📚 Knowledge Base Search

The project includes a knowledge-base tool for retrieving customer-support information including:

* Product specifications
* Return policies
* Warranty information
* Loyalty-program details
* Order-status definitions

### 🌐 Browser Interaction

The project includes AgentCore browser testing through:

```text
test_agentcore_browser.py
```

This provides a browser-oriented demonstration of the agent workflow.

### 💾 AgentCore Memory

Customer interactions can be written to Amazon Bedrock AgentCore Memory and associated with:

```text
Memory
  └── Actor / Customer
        └── Session
              └── Conversation Events
```

### 🧠 Long-Term Customer Preferences

A custom `UserPreferences` memory strategy extracts persistent preferences from conversations.

The demonstrated preferences include:

* Preferred support language → **Marathi**
* Preferred explanation format → **Concise bullet points**

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      Customer        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   AI Support Agent   │
                         │    Amazon Bedrock    │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             ┌────────────┐  ┌────────────┐  ┌──────────────┐
             │ Knowledge  │  │  Browser   │  │   AgentCore  │
             │    Base    │  │ Interaction│  │    Memory    │
             └────────────┘  └────────────┘  └──────┬───────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │ Conversation     │
                                          │ Events / Session │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │ UserPreferences  │
                                          │     Strategy     │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │ Long-Term Memory │
                                          │     Records      │
                                          └──────────────────┘
```

---

# 🧠 AgentCore Memory

The project uses the following AgentCore Memory resource:

| Component | Value                              |
| --------- | ---------------------------------- |
| Region    | `us-east-1`                        |
| Memory    | `CustomerSupportMemory-sRFaSh6LsL` |
| Strategy  | `UserPreferences-p8JsW9Hfv1`       |
| Actor     | `CUST-MEMORY-001`                  |
| Model     | `amazon.nova-lite-v1:0`            |

The long-term memory namespace follows:

```text
/strategies/{memoryStrategyId}/actors/{actorId}/
```

For the demonstrated customer:

```text
/strategies/UserPreferences-p8JsW9Hfv1/actors/CUST-MEMORY-001/
```

---

# 💬 Session Memory vs 🧠 Long-Term Memory

One of the important parts of this project is demonstrating that these are **not the same thing**.

## 💬 Session / Conversation Memory

Conversation turns are written as AgentCore Memory events.

For example:

```text
USER:
My preferred support language is Marathi.

ASSISTANT:
Understood. I can use Marathi for your support conversations.

USER:
I also prefer concise explanations using bullet points.

ASSISTANT:
Understood. I will keep explanations concise and use bullet points.
```

The event is associated with:

```text
Actor ID
CUST-MEMORY-001

Session ID
<conversation-specific session>
```

The event can then be verified through the AgentCore Memory API.

---

## 🧠 Long-Term Memory

The `UserPreferences` strategy processes conversation information and produces persistent preference records.

The successful retrieval produced **two memory records**:

### 🇮🇳 Language Preference

```text
Preferred support language is Marathi.
```

### 📝 Response Style Preference

```text
Prefers concise explanations using bullet points.
```

The records were returned with metadata including:

* Memory record ID
* Memory strategy ID
* Namespace
* Creation timestamp
* Retrieval score
* Record type

This demonstrates the transition from raw conversation events to structured long-term customer preferences.

---

# 🔄 Memory Lifecycle

```text
Customer Conversation
        │
        ▼
   Create Event
        │
        ▼
 AgentCore Memory
        │
        ▼
 UserPreferences
    Extraction
        │
        ▼
 Preference Records
        │
        ▼
 Long-Term Memory
        │
        ▼
 Retrieve for Future
    Personalization
```

The project also configures memory consolidation using three operations:

```text
AddMemory
UpdateMemory
SkipMemory
```

This allows the memory strategy to determine whether a preference is new, should update existing information, or should not be retained.

---

# 📸 Project Evidence

The repository contains screenshots documenting the project and its AWS/AgentCore workflow.

### Deployment

![Deployment](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/deployment_running_sucessfully.png)

### AgentCore / Udacity Evidence

![AgentCore](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/udacity_agentcore.png)

![AgentCore 2](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/udacity_agentcore_2.png)

### Memory Evidence

![Memory 1](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/memory1.png)

![Memory 2](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/memory2.png)

### Session Evidence

![Session](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/session1.png)

### Additional Evidence

![Evidence 1](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/evidence1.png)

![Evidence 2](https://raw.githubusercontent.com/dishapure/AIML-Scholars/main/evidence2.png)

---

# 📁 Project Structure

```text
AIMLSCHOLAR P2/
│
├── main.py
├── invoke_agent.py
├── setup_permissions.py
│
├── test_agentcore_browser.py
├── test_memory_direct.py
├── test_memory_retrieve.py
│
├── product_catalog.md
│
├── agentcore-arm-test.json
├── .bedrock_agentcore.yaml
├── .dockerignore
│
├── pyproject.toml
├── requirements.txt
├── uv.lock
│
├── written_reflection.md
│
└── Screenshots/
```

---

# 📄 File Guide

| File                        | Purpose                         |
| --------------------------- | ------------------------------- |
| `main.py`                   | Main customer-support agent     |
| `invoke_agent.py`           | Agent invocation/testing        |
| `setup_permissions.py`      | AWS permission/setup utilities  |
| `test_agentcore_browser.py` | Browser-based AgentCore testing |
| `test_memory_direct.py`     | Direct AgentCore Memory testing |
| `test_memory_retrieve.py`   | Long-term memory retrieval      |
| `product_catalog.md`        | Product/support information     |
| `agentcore-arm-test.json`   | AgentCore test configuration    |
| `.bedrock_agentcore.yaml`   | AgentCore configuration         |
| `pyproject.toml`            | Python project configuration    |
| `requirements.txt`          | Python dependencies             |
| `uv.lock`                   | Locked dependency versions      |
| `written_reflection.md`     | Project reflection              |
| `Screenshots/`              | Project evidence                |

---

# ⚙️ Technology Stack

| Technology                   | Role                                     |
| ---------------------------- | ---------------------------------------- |
| **Python**                   | Application development                  |
| **Amazon Bedrock**           | Foundation-model powered AI              |
| **Amazon Bedrock AgentCore** | Agent runtime and memory capabilities    |
| **Amazon Nova Lite**         | Memory extraction/consolidation model    |
| **boto3**                    | AWS SDK integration                      |
| **uv**                       | Python environment/dependency management |
| **AWS IAM**                  | Permissions and access control           |

---

# 🔐 IAM & AWS Configuration

The memory configuration uses an execution role:

```text
CustomerSupportMemoryRole
```

The configured foundation-model policy allows:

```text
bedrock:InvokeModel
bedrock:InvokeModelWithResponseStream
```

for:

```text
amazon.nova-lite-v1:0
```

AWS resources used by this project are configured for:

```text
us-east-1
```

> **Security note:** AWS credentials, access keys, secrets, `.env` files, and other sensitive information should never be committed to the repository.

---

# 🛠️ Setup

## 1. Clone the repository

```bash
git clone https://github.com/dishapure/AIML-Scholars.git
cd AIML-Scholars/AIMLSCHOLAR\ P2
```

## 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

## 3. Configure AWS credentials

Configure your AWS credentials using your preferred secure AWS authentication method.

For example:

```bash
aws configure
```

Make sure the configured identity has the permissions required by the project.

---

# ▶️ Running the Project

Run the main application according to the configured AgentCore deployment.

For local Python execution:

```bash
uv run python main.py
```

To invoke/test the agent:

```bash
uv run python invoke_agent.py
```

---

# 🧪 Testing

## Browser Test

```bash
uv run python test_agentcore_browser.py
```

---

## Memory Ingestion Test

```bash
uv run python test_memory_direct.py
```

---

## Memory Retrieval Test

```bash
uv run python test_memory_retrieve.py
```

A successful long-term memory retrieval produces records representing the customer's stored preferences.

---

# 📊 Verified Memory Result

The project successfully produced:

```text
FOUND: 2 MEMORIES
```

The retrieved records represented:

```text
1. Preferred support language is Marathi.

2. Prefers concise explanations using bullet points.
```

The records were associated with:

```text
UserPreferences-p8JsW9Hfv1
```

and:

```text
CUST-MEMORY-001
```

This provides concrete evidence of persistent preference storage and retrieval.

---

# 🌟 Why This Project Matters

Customer-support agents become significantly more useful when they can retain relevant context beyond a single interaction.

This project explores that capability through AgentCore Memory by moving from:

```text
"What did the customer just say?"
```

toward:

```text
"What useful preferences should the system remember
about this customer?"
```

The demonstrated memory pipeline extracts useful preferences such as language and response style and makes those preferences available as structured long-term records.

---

# 🎯 Project Highlights

* 🤖 AI-powered customer-support agent
* ☁️ Amazon Bedrock integration
* 🧠 AgentCore Memory
* 💬 Session-based conversation events
* 🔐 Customer-specific memory
* 📝 Long-term preference extraction
* 🔄 Memory consolidation
* 📚 Knowledge-base search
* 🌐 Browser-based agent testing
* 🧪 Direct AWS API verification
* 📸 Evidence-driven testing
* 🐍 Python + boto3
* ⚡ `uv` development workflow

---

# 📚 Learning Outcomes

Through this project, the implementation demonstrates practical experience with:

* Building AI agents using Amazon Bedrock
* Integrating AWS services using `boto3`
* Working with AgentCore Memory
* Separating actors and sessions
* Creating and inspecting memory events
* Configuring memory strategies
* Extracting persistent user preferences
* Retrieving structured long-term memories
* Working with IAM permissions
* Testing cloud-based AI workflows
* Documenting and validating an end-to-end AI application

---

# 🏆 Project Evidence

The repository includes deployment, browser, session, memory, and AgentCore evidence screenshots.

The most important demonstrated result is the successful retrieval of persistent customer preferences:

> 🇮🇳 **Preferred support language is Marathi.**

> 📝 **Prefers concise explanations using bullet points.**

These records were successfully stored under the configured `UserPreferences` strategy and retrieved from the customer's AgentCore Memory namespace.

---

# 👩‍💻 Author

**Disha Pure**

AI/ML Scholar · AI Agents · AWS · Generative AI

🔗 **GitHub:**
https://github.com/dishapure

---

<p align="center">

### Built with ☁️ AWS + 🤖 AI + 🧠 Agent Memory

**AIML Scholars — Project 2**

</p>
