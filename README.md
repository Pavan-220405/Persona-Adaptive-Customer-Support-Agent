# Persona-Adaptive Customer Support Agent

A modular, latency-optimized, retrieval-augmented customer support agent built using **LangGraph**, **LLMs**, and a **Chroma vector database**.  
The system intelligently adapts responses based on the user's persona, decides when human escalation is required, and retrieves relevant knowledge when necessary.

This project demonstrates how to build a **production-style AI support agent** with:

- Persona-aware responses
- Retrieval-Augmented Generation (RAG)
- Structured LLM outputs
- Pydantic-validated state
- Multi-query retrieval
- Persistent graph state using SQLite
- Latency-optimized architecture

---

# Features

## Persona Adaptive Responses
The system detects the user's persona and adapts its response style accordingly.

Supported personas include:
- **technical_expert**
- **frustrated_user**
- **business_executive**

Persona classification ensures responses match the user's technical depth and tone.

---

## Escalation Detection
The agent automatically decides whether a query should be escalated to human support.

Examples of escalation triggers:
- User frustration
- Billing or account issues
- Requests for refunds or complaints
- Complex issues beyond documentation

If escalation is required, the system routes the query to a **HumanSupport node**.

---

## Retrieval Augmented Generation (RAG)
For knowledge-based queries, the system retrieves relevant documents and generates grounded responses.

Pipeline:
```
Query → Retriever → Context Documents → LLM → Final Answer
```

The retriever uses **ChromaDB** with embeddings to search documentation efficiently.

---

## MultiQuery Retriever
The system uses a **MultiQueryRetriever** to improve retrieval quality.

Instead of searching with a single query, the retriever:
1. Generates multiple reformulated versions of the question
2. Searches the vector database for each variation
3. Combines the retrieved documents

This improves recall and ensures relevant documents are not missed.

---

## Latency Optimized Triage
The initial version of the chatbot required **multiple LLM calls**:

1. Persona detection
2. Escalation decision
3. Retrieval decision

This increased latency.

The optimized architecture introduces a **Triage Node** that performs all decisions in **one structured LLM call**.

The triage output includes:
- Persona
- Confidence score
- Escalation decision
- Retrieval requirement

This significantly reduces latency and token usage.

---

## Structured LLM Outputs
The system uses **structured outputs** instead of free text responses.

Outputs are validated using Pydantic schemas such as:

- `PersonaClassification`
- `EscalationDecision`
- `RetrievalDecision`
- `TriageResult`

This ensures reliable parsing and robust routing logic.

---

## Pydantic State Validation
All LangGraph state is validated using **Pydantic models**.

Example state variables include:

- query
- persona
- chat_history
- context
- answer
- escalation decision
- retrieval decision

Using Pydantic ensures type safety and prevents runtime errors.

---

## SQLite Checkpointer
The system includes an **SQLite-based checkpointer** that persists graph state.

Benefits:
- Resumable executions
- Debugging capability
- Conversation persistence
- Reproducible runs

The checkpointer uses **LangGraph's SqliteSaver**.

Checkpoint files are stored in:

```
checkpoints/langgraph.db
```

---

## Modular Architecture
The project is organized into modular components to improve maintainability.

Key modules include:

- configuration
- prompts
- schemas
- retriever
- data ingestion
- graph logic
- agent implementation

Each module has a clearly defined responsibility.

---

# Project Structure

```
Customer_Support_Chatbot
│
├── my_project
│   │
│   ├── checkpoints
│   │   └── langgraph.db
│   │
│   ├── files
│   │   └── documents for ingestion
│   │
│   ├── rag
│   │
│   │   ├── core
│   │   │   ├── config.py
│   │   │   ├── prompts.py
│   │   │   ├── schemas.py
│   │   │   └── sqlite_checkpointer.py
│   │   │
│   │   ├── graph
│   │   │   ├── agent.py
│   │   │   └── chatbot1.py
│   │   │
│   │   └── load_retrieve
│   │       ├── data_ingestion.py
│   │       └── retriever.py
│   │
│   └── vectorstore
│       └── chroma.sqlite3
│
└── myenv
```

---

# System Architecture

## Original Architecture (Multiple LLM Calls)

```
START
  │
PersonaDetection
  │
EscalationDecision
  │
RetrievalDecision
 ├── Retrieve → Answer
 ├── GeneralAnswer
 └── HumanSupport
  │
 END
```

This design required multiple LLM calls, increasing latency.

---

## Optimized Architecture (Single Triage Call)

```
START
  │
TriageNode
 ├── Retrieve → Answer
 ├── GeneralAnswer
 └── HumanSupport
  │
 END
```

The triage node determines:
- persona
- escalation
- retrieval requirement

This reduces API calls and improves performance.

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <repo-url>
cd Customer_Support_Chatbot
```

---

## 2. Create Virtual Environment

```bash
python -m venv myenv
```

Activate environment:

Windows

```bash
myenv\Scripts\activate
```

Linux / Mac

```bash
source myenv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file with your API keys.

Example:

```
GOOGLE_API_KEY=your_key
HUGGINGFACE_API_KEY=your_key
```

---

## 5. Ingest Documents

Run the ingestion script to create the vector database.

```bash
python rag/load_retrieve/data_ingestion.py
```

This will:

- Load documents from `/files`
- Split them into chunks
- Generate embeddings
- Store them in ChromaDB

---

## 6. Run the Agent

```bash
python rag/graph/agent.py
```

This starts the LangGraph chatbot pipeline.

---

# Core Components

## config.py
Central configuration for:

- model selection
- vector database paths
- checkpoint paths
- file locations

---

## prompts.py
Contains all prompt templates including:

- persona detection
- escalation detection
- retrieval decision
- triage prompt
- answer generation

---

## schemas.py
Defines Pydantic models for:

- structured LLM outputs
- LangGraph state
- triage results
- persona classification

---

## sqlite_checkpointer.py
Creates an SQLite checkpointer for LangGraph using:

```
SqliteSaver
```

Ensures graph state persistence.

---

## agent.py
Main optimized LangGraph agent.

Implements:
- Triage node
- Retrieve node
- Answer node
- GeneralAnswer node
- HumanSupport node

This version minimizes LLM calls.

---

## chatbot1.py
Earlier reference implementation.

Uses multiple nodes:
- PersonaDetection
- EscalationDecision
- RetrievalDecision

Kept for comparison with the optimized design.

---

## retriever.py
Creates the **MultiQueryRetriever** using:

- Chroma vector store
- embedding models
- LLM query expansion

---

## data_ingestion.py
Responsible for:

- loading documents
- splitting into chunks
- embedding generation
- storing in ChromaDB

---

# Why This Project Matters

This project demonstrates several **production-grade AI system patterns**:

- LangGraph workflow orchestration
- Retrieval-Augmented Generation
- Persona-aware conversational agents
- Structured LLM outputs
- Pydantic validation
- Persistent state with checkpointers
- Latency optimization strategies
- Modular AI system design

These patterns are commonly used in **enterprise AI support systems**.

---

# Future Improvements

Possible extensions include:

- HumanSupport based on requirements
- Websearch Fallback
- Streaming responses
- Tool calling
- Conversation memory summarization
- Analytics and monitoring
- Vector database alternatives
- Hybrid search (BM25 + embeddings)
- UI interface (Streamlit or FastAPI)

---

# License

This project is intended for educational and research purposes.
