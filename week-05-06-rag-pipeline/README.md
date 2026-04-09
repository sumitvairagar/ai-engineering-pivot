# Week 5–7: RAG Pipeline — Financial Domain (Vector + Vectorless)

> **Theme:** Build TWO RAG systems on the same financial data — first with vector embeddings (the foundation), then with vectorless reasoning-based retrieval (the cutting edge). Compare them head-to-head. This is the most portfolio-valuable project in the entire plan.
> **Time commitment:** ~2 hrs/day — this is the most hands-on phase
> **Status:** ⬜ Not Started

---

## 🎬 Demoable Deliverable

**What to show:** Two RAG approaches on the same brokerage trade data — vector-based and vectorless — with a side-by-side comparison showing where each wins.

**Demo format:**
- Screen recording or live demo: ask the same 4–5 questions to both systems, compare answers
- Show the Swagger UI at `/ask` endpoint with a `method` toggle (vector vs vectorless)
- A simple HTML chat UI that lets you switch between approaches

**Where to share:** GitHub repo with full setup instructions + a `COMPARISON.md` showing results. This becomes a LinkedIn post: "I built the same financial RAG system two ways — here's what I learned."

---

## 🎯 What You're Building

A system where you can ask questions like:
- *"What were the fill rates for broker A orders over 5000 units last week?"*
- *"Summarize the risk exposure for stop orders during high volatility periods."*
- *"Which order types had the highest rejection rate in December?"*

And get accurate, LLM-generated answers grounded in your actual data — not hallucinations.

**You'll build this twice:**
1. **Vector RAG (Week 5):** Chunk → Embed → Vector search → LLM. The standard approach.
2. **Vectorless RAG (Week 6):** Document tree → LLM reasons over structure → retrieves by logic, not similarity.
3. **Compare + Ship (Week 7):** Head-to-head evaluation, API, Docker, documentation.

---

## 🏗️ Architecture — Two Approaches

### Approach 1: Vector RAG
```
User Question
     ↓
[Embedding model] → converts question to vector
     ↓
[Qdrant] ← stores vectorized trade report chunks
     ↓ (top-k by cosine similarity)
[LLM] ← question + retrieved chunks
     ↓
Answer (grounded in similar text)
```

### Approach 2: Vectorless RAG (PageIndex / Reasoning-Based)
```
User Question
     ↓
[Document Tree] ← hierarchical structure of trade reports
     ↓
[LLM] ← reasons over tree structure to find relevant sections
     ↓ (navigates to exact nodes)
[LLM] ← question + structurally-retrieved content
     ↓
Answer (grounded in logically-relevant sections)
```

**Key difference:** Vector RAG finds text that *sounds similar*. Vectorless RAG finds text that *logically should contain the answer*. Both have tradeoffs — you'll learn when each wins.

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

### 📄 Get Real Financial Data (Day 2)

Instead of synthetic data, you'll use real public market data. This makes your portfolio reproducible and credible.

- [ ] Install data packages:
  ```bash
  pip install yfinance
  ```
- [ ] Create `src/fetch_market_data.py`:

```python
"""
Fetch real market data and generate trade-style report documents.
Uses yfinance for public stock data — no NDA, fully reproducible.
"""
import yfinance as yf
import json
import random
from datetime import datetime

random.seed(42)

# Fetch real stock data for 3 tickers (simulating 3 "brokers")
tickers = {"AAPL": "Broker-A", "MSFT": "Broker-B", "JPM": "Broker-C"}
reports = []

for ticker, broker in tickers.items():
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    for date, row in hist.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        volatility = round(abs(row["High"] - row["Low"]) / row["Open"] * 100, 2)
        volume = int(row["Volume"])

        for order_type in ["market", "limit", "stop"]:
            fill_rate = round(random.uniform(0.7, 0.98) if volatility < 2.0
                            else random.uniform(0.4, 0.75), 2)
            avg_size = random.randint(500, 8000)

            reports.append({
                "id": f"RPT-{date_str}-{broker}-{order_type}",
                "date": date_str,
                "broker": broker,
                "ticker": ticker,
                "order_type": order_type,
                "total_orders": random.randint(50, 500),
                "avg_order_size": avg_size,
                "fill_rate": fill_rate,
                "rejection_rate": round(1 - fill_rate, 2),
                "market_volatility": volatility,
                "close_price": round(row["Close"], 2),
                "volume": volume,
                "risk_score": round(volatility * (1 - fill_rate), 2),
                "summary": (
                    f"{broker} ({ticker}) processed {random.randint(50,500)} {order_type} "
                    f"orders on {date_str}. Close price: ${row['Close']:.2f}. "
                    f"Intraday volatility: {volatility}%. Fill rate: {fill_rate*100:.1f}%. "
                    f"Volume: {volume:,} shares. Avg order size: {avg_size} units. "
                    f"{'High volatility impacted fills.' if volatility > 2.0 else 'Stable conditions.'} "
                    f"Risk score: {round(volatility * (1-fill_rate), 2)}."
                )
            })

with open("src/trade_reports.json", "w") as f:
    json.dump(reports, f, indent=2)

print(f"Generated {len(reports)} trade reports from real market data")
print(f"Date range: {reports[0]['date']} to {reports[-1]['date']}")
print(f"Sample: {reports[0]['summary']}")
```

- [ ] Run it: `python src/fetch_market_data.py`
- [ ] Open `trade_reports.json` — these are based on real AAPL, MSFT, JPM price data
- [ ] **Key insight:** Real data has real patterns — earnings days spike volatility, volumes cluster around market open/close. Your model and RAG will pick up on these.

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

## ✅ Week 6 Checklist — Vectorless RAG (Reasoning-Based Retrieval)

### 📖 Understand Why Vectorless Exists (Day 1)
- [ ] Read: [Microsoft — Vectorless Reasoning-Based RAG](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/vectorless-reasoning-based-rag-a-new-approach-to-retrieval-augmented-generation/4502238) (~20 min)
- [ ] Read: [PageIndex: The Vectorless RAG](https://pvtech.substack.com/p/pageindex-the-vectorless-rag) (~15 min)
- [ ] **What to absorb:** Vector RAG chunks documents and searches by similarity — it can miss context split across chunks and retrieve text that "sounds right" but isn't logically relevant. Vectorless RAG keeps document structure intact and uses the LLM to reason about WHERE the answer should be. Think of it like: vector search = full-text search on steroids, vectorless = a human reading a table of contents.

### 🌳 Build a Document Tree from Your Trade Reports (Day 2–3)

Instead of chunking and embedding, you'll organize your trade reports into a navigable tree structure that an LLM can reason over.

- [ ] Create `src/build_document_tree.py`:

```python
"""
Build a hierarchical document tree from trade reports.
Instead of embedding chunks, we organize data by structure:
  Root → Broker → Order Type → Monthly summaries → Daily reports

This is what the LLM will "navigate" to find answers.
"""
import json
from collections import defaultdict
from datetime import datetime

with open("src/trade_reports.json") as f:
    reports = json.load(f)

# Build tree: Broker → Order Type → Month → Reports
tree = {"title": "Trade Reports Q4 2024", "children": [], "node_id": "root"}
broker_nodes = {}

for report in reports:
    broker = report["broker"]
    order_type = report["order_type"]
    month = datetime.strptime(report["date"], "%Y-%m-%d").strftime("%Y-%m")

    # Broker level
    if broker not in broker_nodes:
        broker_node = {
            "node_id": f"broker-{broker}",
            "title": f"Broker {broker}",
            "summary": f"All trade reports for Broker {broker}",
            "children": {}
        }
        broker_nodes[broker] = broker_node
        tree["children"].append(broker_node)

    # Order type level
    bn = broker_nodes[broker]
    if order_type not in bn["children"]:
        bn["children"][order_type] = {
            "node_id": f"broker-{broker}-{order_type}",
            "title": f"Broker {broker} — {order_type} orders",
            "summary": f"{order_type.title()} order reports for Broker {broker}",
            "children": {}
        }

    # Month level
    ot_node = bn["children"][order_type]
    if month not in ot_node["children"]:
        ot_node["children"][month] = {
            "node_id": f"broker-{broker}-{order_type}-{month}",
            "title": f"{order_type.title()} orders — {month}",
            "summary": "",
            "reports": []
        }

    ot_node["children"][month]["reports"].append(report)

# Compute monthly summaries
for broker_node in tree["children"]:
    for ot_key, ot_node in broker_node["children"].items():
        for month_key, month_node in ot_node["children"].items():
            reps = month_node["reports"]
            avg_fill = sum(r["fill_rate"] for r in reps) / len(reps)
            avg_vol = sum(r["market_volatility"] for r in reps) / len(reps)
            month_node["summary"] = (
                f"{len(reps)} reports. Avg fill rate: {avg_fill:.1%}. "
                f"Avg volatility: {avg_vol:.2f}. "
                f"Date range: {reps[0]['date']} to {reps[-1]['date']}."
            )

# Convert nested dicts to lists for clean JSON
def flatten_children(node):
    if "children" in node and isinstance(node["children"], dict):
        node["children"] = list(node["children"].values())
        for child in node["children"]:
            flatten_children(child)

flatten_children(tree)

with open("src/document_tree.json", "w") as f:
    json.dump(tree, f, indent=2)

# Print tree structure
def print_tree(node, indent=0):
    prefix = "  " * indent
    title = node.get("title", "?")
    summary = node.get("summary", "")[:60]
    print(f"{prefix}├── {title} | {summary}")
    for child in node.get("children", []):
        print_tree(child, indent + 1)

print_tree(tree)
print(f"\nTree saved to src/document_tree.json")
```

- [ ] Run it — observe the hierarchical structure
- [ ] **Key insight:** This is how PageIndex works conceptually. Instead of "find similar text," the LLM will look at this tree and reason: "The question is about Broker A stop orders → navigate to Broker A → stop orders → relevant month."

### 🤖 Build the Vectorless RAG Chain (Day 4–5)
- [ ] Create `src/vectorless_rag_chain.py`:

```python
"""
Vectorless RAG: LLM reasons over document tree structure
to find relevant content — no embeddings, no vector DB.
"""
import json
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

class VectorlessTradeRAG:
    def __init__(self):
        self.llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        with open("src/document_tree.json") as f:
            self.tree = json.load(f)

    def _tree_without_reports(self):
        """Return tree structure with summaries but without full report text"""
        def strip(node):
            out = {"node_id": node.get("node_id"), "title": node.get("title"),
                   "summary": node.get("summary", "")}
            if "children" in node:
                out["children"] = [strip(c) for c in node["children"]]
            if "reports" in node:
                out["report_count"] = len(node["reports"])
            return out
        return strip(self.tree)

    def _get_node_content(self, node_ids: list) -> str:
        """Retrieve full content for specific nodes"""
        content = []
        def search(node):
            if node.get("node_id") in node_ids:
                if "reports" in node:
                    for r in node["reports"]:
                        content.append(r["summary"])
                elif "summary" in node:
                    content.append(node["summary"])
            for child in node.get("children", []):
                search(child)
        search(self.tree)
        return "\n\n".join(content)

    def answer(self, question: str) -> dict:
        # Step 1: LLM reasons over tree to find relevant nodes
        tree_summary = json.dumps(self._tree_without_reports(), indent=2)

        nav_response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": f"""You are navigating a document tree to find trade reports relevant to a question.

DOCUMENT TREE (summaries only):
{tree_summary}

QUESTION: {question}

Which node_ids contain data needed to answer this question? Think step by step about where the answer would logically be, then return JSON:
{{"reasoning": "your thinking", "node_ids": ["id1", "id2"]}}

Return ONLY the JSON."""}]
        )

        nav_result = json.loads(nav_response.content[0].text)
        reasoning = nav_result.get("reasoning", "")
        node_ids = nav_result.get("node_ids", [])

        # Step 2: Retrieve content from selected nodes
        context = self._get_node_content(node_ids)

        # Step 3: Generate answer from retrieved content
        answer_response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": f"""You are a financial analyst. Answer using ONLY the data below.
Be specific — reference dates, brokers, and numbers.

RETRIEVED DATA:
{context}

QUESTION: {question}

ANSWER:"""}]
        )

        return {
            "question": question,
            "answer": answer_response.content[0].text,
            "method": "vectorless",
            "reasoning": reasoning,
            "nodes_retrieved": node_ids,
        }

if __name__ == "__main__":
    rag = VectorlessTradeRAG()
    questions = [
        "What were the fill rates for Broker A during high volatility periods?",
        "Which order type had the highest rejection rate across all brokers?",
        "Compare the risk scores between brokers B and C for stop orders.",
    ]
    for q in questions:
        result = rag.answer(q)
        print(f"\nQ: {q}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Nodes: {result['nodes_retrieved']}")
        print(f"A: {result['answer']}")
        print("-" * 60)
```

- [ ] Run it — observe how the LLM navigates the tree vs how vector search finds chunks
- [ ] Compare the reasoning trace to vector RAG — which found more relevant data?

---

## ✅ Week 7 Checklist — Compare, Ship, and Document

### 📊 Head-to-Head Comparison (Day 1–2)
- [ ] Create `src/compare_rag.py`:

```python
"""
Run the same questions through both RAG approaches and compare.
"""
from src.rag_chain import TradeReportRAG
from src.vectorless_rag_chain import VectorlessTradeRAG
import time
import json

vector_rag = TradeReportRAG()
vectorless_rag = VectorlessTradeRAG()

questions = [
    "What were the fill rates for Broker A during high volatility periods?",
    "Which order type had the highest rejection rate across all brokers?",
    "Compare the risk scores between brokers B and C for stop orders.",
    "Were there any trends in market volatility over the last 3 months?",
    "What was Broker C's performance on limit orders in November?",
]

results = []
for q in questions:
    # Vector RAG
    start = time.time()
    v_result = vector_rag.answer(q)
    v_time = (time.time() - start) * 1000

    # Vectorless RAG
    start = time.time()
    vl_result = vectorless_rag.answer(q)
    vl_time = (time.time() - start) * 1000

    results.append({
        "question": q,
        "vector": {"answer": v_result["answer"], "latency_ms": round(v_time)},
        "vectorless": {
            "answer": vl_result["answer"],
            "reasoning": vl_result["reasoning"],
            "latency_ms": round(vl_time),
        },
    })

    print(f"\nQ: {q}")
    print(f"  VECTOR ({v_time:.0f}ms): {v_result['answer'][:120]}...")
    print(f"  VECTORLESS ({vl_time:.0f}ms): {vl_result['answer'][:120]}...")

with open("src/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to src/comparison_results.json")
```

- [ ] Run the comparison — note which approach gives better answers for which question types
- [ ] Write `COMPARISON.md` documenting your findings:
  - Where vector RAG won (broad semantic queries)
  - Where vectorless won (structured/specific queries)
  - Latency differences (vectorless uses 2 LLM calls vs 1)
  - Cost implications

### 🌐 Unified API + UI (Day 3–4)
- [ ] Create `src/rag_api.py` — FastAPI wrapper with both approaches:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_chain import TradeReportRAG
from src.vectorless_rag_chain import VectorlessTradeRAG
import time

app = FastAPI(title="Trade Intelligence RAG API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

vector_rag = TradeReportRAG()
vectorless_rag = VectorlessTradeRAG()

class Question(BaseModel):
    question: str
    method: str = "vector"  # "vector" or "vectorless"

class Answer(BaseModel):
    question: str
    answer: str
    method: str
    latency_ms: float
    metadata: dict = {}

@app.post("/ask", response_model=Answer)
async def ask(request: Question):
    start = time.time()
    rag = vector_rag if request.method == "vector" else vectorless_rag
    result = rag.answer(request.question)
    return Answer(
        question=result["question"],
        answer=result["answer"],
        method=request.method,
        latency_ms=round((time.time() - start) * 1000, 1),
        metadata={"reasoning": result.get("reasoning", ""),
                   "nodes": result.get("nodes_retrieved", []),
                   "sources_used": result.get("sources_used", 0)},
    )

@app.get("/health")
async def health():
    return {"status": "UP", "methods": ["vector", "vectorless"]}
```

- [ ] Run: `uvicorn src.rag_api:app --reload --port 8001`
- [ ] Test both methods via Swagger at http://localhost:8001/docs
- [ ] Create a simple `index.html` chat UI with a toggle to switch between vector and vectorless

### 🎯 Evaluation (Day 5)
- [ ] Create `src/evaluate_rag.py` — test both approaches against the same eval set:

```python
from src.rag_chain import TradeReportRAG
from src.vectorless_rag_chain import VectorlessTradeRAG

eval_set = [
    {"question": "What is the fill rate for Broker A market orders?",
     "expected_contains": ["broker A", "market", "fill rate"]},
    {"question": "Which broker has the highest risk score?",
     "expected_contains": ["risk", "broker"]},
    {"question": "How did Broker C stop orders perform in November 2024?",
     "expected_contains": ["broker C", "stop", "november"]},
]

for name, rag in [("VECTOR", TradeReportRAG()), ("VECTORLESS", VectorlessTradeRAG())]:
    print(f"\n{'='*50}\n{name} RAG Evaluation\n{'='*50}")
    for item in eval_set:
        result = rag.answer(item["question"])
        answer_lower = result["answer"].lower()
        hits = [kw for kw in item["expected_contains"] if kw.lower() in answer_lower]
        score = len(hits) / len(item["expected_contains"])
        print(f"\nQ: {item['question']}")
        print(f"  Keywords: {hits}/{item['expected_contains']} → {score:.0%}")
```

- [ ] Run evaluation, compare grounding scores between approaches

### 📝 Docker + Final Polish (Day 6–7)
- [ ] Create `Dockerfile.rag` for the unified RAG API
- [ ] Update `docker-compose.yml` to include Qdrant + RAG API together
- [ ] Write `ARCHITECTURE.md` with diagrams of BOTH RAG approaches
- [ ] Write `src/README.md` with:
  - What it does and why two approaches
  - How each RAG pipeline works (explain simply)
  - How to run it
  - Example questions and answers from both approaches
  - Your comparison findings
- [ ] Commit everything
- [ ] Update main README → Week 5–7 ✅ Done

---

## 📚 Resource Reference

| Resource | Type | Time | Priority |
|----------|------|------|----------|
| [LangChain RAG Concepts](https://python.langchain.com/docs/concepts/rag/) | Docs | 20 min | 🔴 Must |
| [What is a Vector DB](https://qdrant.tech/articles/what-is-a-vector-database/) | Article | 15 min | 🔴 Must |
| [Qdrant Quickstart](https://qdrant.tech/documentation/quick-start/) | Docs | 20 min | 🔴 Must |
| [Sentence Transformers](https://www.sbert.net/docs/quickstart.html) | Docs | 20 min | 🔴 Must |
| [Microsoft — Vectorless RAG](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/vectorless-reasoning-based-rag-a-new-approach-to-retrieval-augmented-generation/4502238) | Blog | 20 min | 🔴 Must |
| [PageIndex: The Vectorless RAG](https://pvtech.substack.com/p/pageindex-the-vectorless-rag) | Blog | 15 min | 🔴 Must |
| [Context Engineering: The Hidden Discipline](https://stepto.net/blog/context-engineering-ai-agents-production-2026) | Blog | 25 min | 🔴 Must |
| [Context Engineering for Teams — Taskade](https://www.taskade.com/blog/context-engineering-teams-guide) | Guide | 20 min | 🟡 Recommended |
| [All you need to know about RAG in 2026](https://open.substack.com/pub/aishwaryasrinivasan/p/all-you-need-to-know-about-rag-in) | Blog | 20 min | 🟡 Recommended |
| [RAGAS Evaluation](https://docs.ragas.io/en/latest/) | Docs | 20 min | 🟡 Recommended |
| [RAG from Scratch — LangChain YouTube](https://www.youtube.com/watch?v=wd7TZ4w1mSw) | Video | 45 min | 🟡 Recommended |
| [LangChain State of AI Agents 2026](https://www.langchain.com/state-of-agent-engineering) | Report | 30 min | 🟡 Recommended |

---

## 🏁 End-of-Phase Self-Assessment

- [ ] I have TWO working RAG systems on the same financial data
- [ ] I understand embeddings, semantic search, and why they sometimes fail
- [ ] I understand vectorless/reasoning-based retrieval and when it's better
- [ ] I can articulate the tradeoffs between both approaches to a technical audience
- [ ] My unified RAG API is containerized and documented with comparison results
- [ ] I have a `COMPARISON.md` that shows real findings — this becomes a LinkedIn post

**Next:** [Week 8–9 — Agent Patterns + Agentic AI →](../week-07-08-agentic-ai/README.md)
