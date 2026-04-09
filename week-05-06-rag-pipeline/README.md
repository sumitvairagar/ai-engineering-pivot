# Week 5–6: RAG Pipeline — Financial Domain

> **Theme:** Build a Retrieval-Augmented Generation (RAG) system that answers questions about order/trade data using LLMs. This is the most portfolio-valuable project in the entire plan.
> **Time commitment:** ~2 hrs/day — this is the most hands-on phase
> **Status:** ⬜ Not Started

---

## 🎯 What You're Building

A system where you can ask questions like:
- *"What were the fill rates for broker A orders over 5000 units last week?"*
- *"Summarize the risk exposure for stop orders during high volatility periods."*
- *"Which order types had the highest rejection rate in December?"*

And get accurate, LLM-generated answers grounded in your actual data — not hallucinations.

**Stack:** LangChain + Qdrant (vector DB) + Claude/OpenAI API + FastAPI

---

## 🏗️ Architecture

```
User Question
     ↓
[FastAPI endpoint]
     ↓
[Embedding model] → converts question to vector
     ↓
[Qdrant] ← stores vectorized order/trade documents
     ↓ (top-k similar docs)
[LLM (Claude/GPT)] ← question + retrieved context
     ↓
Answer grounded in real data
```

This is RAG. The key insight: **LLM doesn't need to memorize your data — it reasons over documents retrieved at query time.**

---

## ✅ Week 5 Checklist

### 🔧 Setup (Day 1)
- [ ] Install dependencies:
  ```bash
  pip install langchain langchain-community langchain-openai \
              qdrant-client sentence-transformers \
              anthropic openai python-dotenv
  ```
- [ ] Create `.env` file (never commit this):
  ```
  OPENAI_API_KEY=your_key_here
  # OR
  ANTHROPIC_API_KEY=your_key_here
  ```
- [ ] Read: [LangChain RAG — Conceptual Guide](https://python.langchain.com/docs/concepts/rag/) (~20 min) — understand: documents, embeddings, vector store, retriever, chain
- [ ] Read: [What is a Vector Database](https://qdrant.tech/articles/what-is-a-vector-database/) (~15 min)
- [ ] **What to absorb:** An embedding is a list of ~1500 numbers that captures the *meaning* of a text. Similar texts have similar numbers. A vector DB lets you search by meaning, not keywords.

### 📄 Generate Synthetic Financial Documents (Day 2)
- [ ] Create `src/generate_docs.py`:

```python
"""
Generate synthetic trade/order documents that simulate 
what you'd have in a real brokerage system like Manafsoft.
Each "document" is a trade summary report.
"""
import json
import random
from datetime import datetime, timedelta

random.seed(42)

def generate_trade_report(date, broker, order_type, size_range, volatility_range):
    size = random.randint(*size_range)
    volatility = round(random.uniform(*volatility_range), 2)
    fill_rate = round(random.uniform(0.55, 0.98) if volatility < 1.2 else random.uniform(0.3, 0.7), 2)
    
    return {
        "id": f"REPORT-{date.strftime('%Y%m%d')}-{broker}-{random.randint(1000,9999)}",
        "date": date.strftime("%Y-%m-%d"),
        "broker": broker,
        "order_type": order_type,
        "total_orders": random.randint(50, 500),
        "avg_order_size": size,
        "fill_rate": fill_rate,
        "rejection_rate": round(1 - fill_rate, 2),
        "market_volatility": volatility,
        "volume_traded": size * random.randint(50, 500),
        "risk_score": round(volatility * (1 - fill_rate) * 10, 2),
        "summary": (
            f"Broker {broker} processed {random.randint(50,500)} {order_type} orders "
            f"on {date.strftime('%B %d, %Y')}. Average order size was {size} units. "
            f"Fill rate was {fill_rate*100:.1f}%. Market volatility index was {volatility}. "
            f"{'High volatility impacted fill rates significantly.' if volatility > 1.5 else 'Market conditions were stable.'} "
            f"Risk score: {round(volatility * (1-fill_rate) * 10, 2)}/10."
        )
    }

# Generate 90 days of reports
reports = []
base_date = datetime(2024, 10, 1)

for day_offset in range(90):
    date = base_date + timedelta(days=day_offset)
    if date.weekday() >= 5:  # skip weekends
        continue
    for broker in ["A", "B", "C"]:
        for order_type in ["market", "limit", "stop"]:
            size_range = {"A": (3000, 8000), "B": (1000, 4000), "C": (500, 2000)}[broker]
            vol_range = (0.5, 2.5) if day_offset > 60 else (0.3, 1.5)  # market changes
            reports.append(generate_trade_report(date, broker, order_type, size_range, vol_range))

# Save
with open("src/trade_reports.json", "w") as f:
    json.dump(reports, f, indent=2)

print(f"Generated {len(reports)} trade reports")
print(f"Sample: {reports[0]['summary']}")
```

- [ ] Run it: `python src/generate_docs.py`
- [ ] Open `trade_reports.json`, read a few summaries — they should feel realistic

### 🗂️ Build the Vector Index (Day 3–4)
- [ ] Start Qdrant locally with Docker:
  ```bash
  docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
  ```
- [ ] Create `src/index_documents.py`:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
import uuid

# Load documents
with open("src/trade_reports.json") as f:
    reports = json.load(f)

# Embedding model (runs locally, no API cost)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, 384 dimensions

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Create collection (like creating an index in a DB)
COLLECTION = "trade_reports"
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Embed and index documents
print(f"Indexing {len(reports)} documents...")
points = []
for i, report in enumerate(reports):
    # What we embed: the summary text (this is what gets searched semantically)
    vector = embedder.encode(report["summary"]).tolist()
    
    points.append(PointStruct(
        id=i,
        vector=vector,
        payload={
            "id": report["id"],
            "date": report["date"],
            "broker": report["broker"],
            "order_type": report["order_type"],
            "fill_rate": report["fill_rate"],
            "market_volatility": report["market_volatility"],
            "risk_score": report["risk_score"],
            "summary": report["summary"]
        }
    ))

# Batch upsert
client.upsert(collection_name=COLLECTION, points=points)
print(f"Indexed {len(points)} documents into Qdrant")

# Quick test search
query = "broker A high volatility orders"
query_vec = embedder.encode(query).tolist()
results = client.search(collection_name=COLLECTION, query_vector=query_vec, limit=3)

print(f"\nTest search: '{query}'")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['summary'][:80]}...")
```

- [ ] Run: `python src/index_documents.py`
- [ ] You should see 3 relevant documents retrieved for the test query
- [ ] Try changing the query — observe how semantic search works vs keyword search

### 🤖 Build the RAG Chain (Day 5)
- [ ] Create `src/rag_chain.py`:

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import anthropic  # or use openai
import os
from dotenv import load_dotenv

load_dotenv()

class TradeReportRAG:
    def __init__(self):
        self.client = QdrantClient("localhost", port=6333)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.collection = "trade_reports"
    
    def retrieve(self, query: str, top_k: int = 5):
        """Semantic search — find most relevant trade reports"""
        query_vec = self.embedder.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k
        )
        return [r.payload for r in results]
    
    def answer(self, question: str) -> dict:
        """Full RAG pipeline: retrieve → augment → generate"""
        # 1. Retrieve relevant documents
        docs = self.retrieve(question, top_k=5)
        
        # 2. Build context from retrieved docs
        context = "\n\n".join([
            f"Report [{i+1}] ({d['date']}, Broker {d['broker']}, {d['order_type']}):\n{d['summary']}"
            for i, d in enumerate(docs)
        ])
        
        # 3. Build prompt (augmented with retrieved context)
        prompt = f"""You are a financial analyst assistant for a brokerage operations team.
Answer the question using ONLY the trade report data provided below.
If the data doesn't contain enough information to answer confidently, say so.
Be specific — reference dates, brokers, and numbers from the reports.

TRADE REPORT DATA:
{context}

QUESTION: {question}

ANSWER:"""
        
        # 4. Generate answer with LLM
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text
        
        return {
            "question": question,
            "answer": answer,
            "sources_used": len(docs),
            "source_dates": [d["date"] for d in docs],
            "source_brokers": list(set(d["broker"] for d in docs))
        }

# Test it
if __name__ == "__main__":
    rag = TradeReportRAG()
    
    questions = [
        "What were the fill rates for Broker A during high volatility periods?",
        "Which order type had the highest rejection rate across all brokers?",
        "Compare the risk scores between brokers B and C for stop orders.",
        "Were there any trends in market volatility over the last 3 months of data?",
    ]
    
    print("=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        result = rag.answer(q)
        print(f"A: {result['answer']}")
        print(f"   [Sources: {result['sources_used']} reports from brokers {result['source_brokers']}]")
        print("-" * 60)
```

- [ ] Run: `python src/rag_chain.py`
- [ ] Read the answers carefully — are they grounded in actual data from the reports?
- [ ] Try adding your own questions

---

## ✅ Week 6 Checklist

### 🌐 Expose RAG as API + UI (Day 1–2)
- [ ] Create `src/rag_api.py` — FastAPI wrapper:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_chain import TradeReportRAG
import time

app = FastAPI(title="Trade Intelligence RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = TradeReportRAG()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str
    sources_used: int
    latency_ms: float

@app.post("/ask", response_model=Answer)
async def ask(request: Question):
    start = time.time()
    result = rag.answer(request.question)
    return Answer(
        question=result["question"],
        answer=result["answer"],
        sources_used=result["sources_used"],
        latency_ms=round((time.time() - start) * 1000, 1)
    )

@app.get("/health")
async def health():
    return {"status": "UP", "collection": "trade_reports"}
```

- [ ] Run: `uvicorn src.rag_api:app --reload --port 8001`
- [ ] Test via Swagger at http://localhost:8001/docs
- [ ] Create a simple `index.html` chat UI (even basic HTML/JS is fine) that calls your `/ask` endpoint

### 🎯 Evaluation: Is Your RAG Good? (Day 3–4)
- [ ] Read: [RAG Evaluation Basics](https://docs.ragas.io/en/latest/concepts/metrics/index.html) (~20 min)
- [ ] Create `src/evaluate_rag.py`:

```python
"""
Simple RAG evaluation — measure if answers are grounded in sources
"""
from src.rag_chain import TradeReportRAG

rag = TradeReportRAG()

eval_set = [
    {
        "question": "What is the fill rate for Broker A market orders?",
        "expected_contains": ["broker A", "market", "fill rate"],
    },
    {
        "question": "Which broker has the highest risk score?",
        "expected_contains": ["risk", "broker"],
    },
]

print("RAG Evaluation")
print("=" * 50)
for item in eval_set:
    result = rag.answer(item["question"])
    answer_lower = result["answer"].lower()
    
    hits = [kw for kw in item["expected_contains"] if kw.lower() in answer_lower]
    score = len(hits) / len(item["expected_contains"])
    
    print(f"\nQ: {item['question']}")
    print(f"Keywords found: {hits}/{item['expected_contains']}")
    print(f"Grounding score: {score:.0%}")
    print(f"Sources used: {result['sources_used']}")
```

- [ ] Run evaluation, note the grounding scores
- [ ] Try changing `top_k` in the retrieve method (3 vs 5 vs 8) — see how it affects answer quality

### 📝 Add Docker + Final Polish (Day 5–7)
- [ ] Create `Dockerfile.rag` for the RAG API
- [ ] Update `docker-compose.yml` to include Qdrant + RAG API together
- [ ] Write `ARCHITECTURE.md` with a diagram of the full RAG system
- [ ] Write `src/README.md` with:
  - What it does
  - How the RAG pipeline works (explain it simply)
  - How to run it
  - Example questions and answers (copy your best outputs)
- [ ] Commit everything
- [ ] Update main README → Week 5–6 ✅ Done

---

## 📚 Resource Reference

| Resource | Type | Time | Priority |
|----------|------|------|----------|
| [LangChain RAG Concepts](https://python.langchain.com/docs/concepts/rag/) | Docs | 20 min | 🔴 Must |
| [What is a Vector DB](https://qdrant.tech/articles/what-is-a-vector-database/) | Article | 15 min | 🔴 Must |
| [Qdrant Quickstart](https://qdrant.tech/documentation/quick-start/) | Docs | 20 min | 🔴 Must |
| [Sentence Transformers](https://www.sbert.net/docs/quickstart.html) | Docs | 20 min | 🔴 Must |
| [RAGAS Evaluation](https://docs.ragas.io/en/latest/) | Docs | 20 min | 🟡 Recommended |
| [RAG from Scratch — LangChain YouTube](https://www.youtube.com/watch?v=wd7TZ4w1mSw) | Video | 45 min | 🟡 Recommended |

---

## 🏁 End-of-Phase Self-Assessment

- [ ] I have a working RAG system that answers questions about trade data
- [ ] I understand what embeddings are and how semantic search differs from keyword search
- [ ] I can explain RAG architecture in 2 minutes to a non-technical stakeholder
- [ ] My RAG API is containerized and documented
- [ ] I have evaluated my RAG system and understand its limitations

**Next:** [Week 7–8 — Agent Patterns + Agentic AI →](../week-07-08-agentic-ai/README.md)
