# Week 8–9: Agent Patterns + Agentic AI

> **Theme:** Extend your existing agent knowledge (from your workshop) into production-grade patterns. This phase is 70% already yours — you're formalizing and deepening what you've built.
> **Time commitment:** ~1.5 hrs/day
> **Status:** ⬜ Not Started

---

## 🎬 Demoable Deliverable

**What to show:** A multi-tool brokerage agent that reasons through complex scenarios — querying trade history (RAG), predicting order fills (ML model), and flagging risky orders (compliance tool) — all in one conversation.

**Demo format:**
- Screen recording of the agent handling 2–3 realistic scenarios end-to-end
- Show the tool call chain: agent reasons → calls RAG → calls ML model → synthesizes answer → flags order
- Terminal output showing the full ReAct loop with tool calls and reasoning

**Where to share:** GitHub repo with the agent config + a `BYOA_CONCEPT.md` explaining the methodology. This becomes the foundation for your Week 9 technical article.

---

## 🎯 What You're Building

A multi-tool agent that can:
1. Answer questions about trades (using your RAG pipeline)
2. Run analytical queries (using your ML model)
3. Take actions (simulate: flag orders, send alerts)

This is your **BYOA proof-of-concept** — a domain-specific agent for brokerage operations.

---

## 🗺️ Agent Patterns You'll Cover

| Pattern | What It Is | When to Use |
|---------|-----------|-------------|
| **ReAct** | Reason → Act → Observe loop | General purpose tool use |
| **Tool Use** | Agent calls external functions | Data retrieval, APIs |
| **Multi-step** | Chain multiple reasoning steps | Complex analysis |
| **Memory** | Persist context across turns | Conversational agents |
| **Subagents** | Agent spawning sub-agents | Parallelism, specialization |

You already know the agent loop from your workshop. This phase adds **production patterns** — memory, tool routing, error handling, observability.

---

## ✅ Week 8 Checklist

### 📖 Deepen Agent Theory (Day 1)
- [ ] Read: [Anthropic's Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — the full post (~45 min). This is the canonical reference.
- [ ] Read: [ReAct Paper summary](https://react-lm.github.io/) — just the abstract and examples (~15 min)
- [ ] **Key concepts to absorb:**
  - Why simple agents beat complex ones (Anthropic's key finding)
  - The difference between workflow (fixed) vs agent (dynamic)
  - When NOT to use agents (important!)
- [ ] Write in `notes.md`: Based on the Manafsoft engagement — which workflows would benefit from agents vs which should stay deterministic?

### 🛠️ Build a Multi-Tool Agent (Day 2–3)
- [ ] Create `src/brokerage_agent.py`:

```python
"""
Brokerage Operations Agent
Combines: RAG retrieval + ML prediction + rule-based tools
This is your BYOA prototype for financial domain agents.
"""
import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# TOOL DEFINITIONS
# These are the agent's "capabilities"
# ============================================================

tools = [
    {
        "name": "query_trade_reports",
        "description": (
            "Search historical trade reports and answer questions about fill rates, "
            "rejection rates, broker performance, and market conditions. "
            "Use this for any analytical question about past trade data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The analytical question about trade data"
                },
                "date_range": {
                    "type": "string", 
                    "description": "Optional date range like 'last 30 days' or 'October 2024'"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "predict_order_fill",
        "description": (
            "Predict whether a specific order will be filled based on current conditions. "
            "Use this when given specific order parameters and asked for a prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_size": {"type": "integer", "description": "Order size in units"},
                "broker_tier": {"type": "string", "description": "Broker tier: A, B, or C"},
                "market_volatility": {"type": "number", "description": "Current market volatility (0.1 to 5.0)"},
                "time_of_day_hour": {"type": "integer", "description": "Hour of day (9-16)"},
                "order_type": {"type": "string", "description": "market, limit, or stop"},
                "days_since_last_trade": {"type": "integer", "description": "Days since last trade"}
            },
            "required": ["order_size", "broker_tier", "market_volatility", 
                        "time_of_day_hour", "order_type", "days_since_last_trade"]
        }
    },
    {
        "name": "flag_order_for_review",
        "description": (
            "Flag an order for manual review by the compliance team. "
            "Use this when risk score is high or fill probability is very low."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why this order needs review"},
                "priority": {"type": "string", "description": "HIGH, MEDIUM, or LOW"}
            },
            "required": ["order_id", "reason", "priority"]
        }
    },
    {
        "name": "get_market_summary",
        "description": "Get a summary of current market conditions for decision making.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def query_trade_reports(question: str, date_range: str = None) -> str:
    """Calls your RAG pipeline"""
    import requests
    try:
        r = requests.post("http://localhost:8001/ask", 
                         json={"question": question}, timeout=15)
        if r.ok:
            data = r.json()
            return f"Analysis (from {data['sources_used']} reports): {data['answer']}"
        return "RAG service unavailable. Here's what I know from training: Fill rates typically range 60-90%."
    except:
        return "RAG service not running. [In production: would query trade_reports collection]"

def predict_order_fill(order_size, broker_tier, market_volatility, 
                       time_of_day_hour, order_type, days_since_last_trade) -> str:
    """Calls your FastAPI ML model"""
    import requests
    payload = {
        "order_size": order_size, "broker_tier": broker_tier,
        "market_volatility": market_volatility, "time_of_day_hour": time_of_day_hour,
        "order_type": order_type, "days_since_last_trade": days_since_last_trade
    }
    try:
        r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        if r.ok:
            d = r.json()
            return (f"Prediction: {d['prediction']} | "
                   f"Fill probability: {d['order_fill_probability']} | "
                   f"Confidence: {d['confidence']}")
        return "ML service unavailable"
    except:
        # Fallback heuristic for demo
        prob = 0.75 if order_type == "market" and market_volatility < 1.2 else 0.45
        return f"[Fallback] Estimated fill probability: {prob} based on order type and volatility"

def flag_order_for_review(order_id: str, reason: str, priority: str) -> str:
    """Simulates writing to a compliance system"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "order_id": order_id, "reason": reason, 
        "priority": priority, "flagged_at": timestamp,
        "flagged_by": "BrokerageAgent/v1"
    }
    print(f"\n📋 COMPLIANCE FLAG: {json.dumps(log_entry, indent=2)}")
    return f"Order {order_id} flagged for {priority} review: {reason}"

def get_market_summary() -> str:
    """Simulates a market data API"""
    import random
    random.seed(int(datetime.now().timestamp()) % 100)
    vol = round(random.uniform(0.8, 2.2), 2)
    return (f"Market conditions as of {datetime.now().strftime('%H:%M')}: "
           f"Volatility index: {vol} ({'HIGH' if vol > 1.5 else 'NORMAL'}). "
           f"Market hours: OPEN. Avg fill rate trend: {'declining' if vol > 1.5 else 'stable'}.")

# ============================================================
# TOOL DISPATCHER
# ============================================================

def execute_tool(tool_name: str, tool_input: dict) -> str:
    handlers = {
        "query_trade_reports": lambda i: query_trade_reports(**i),
        "predict_order_fill": lambda i: predict_order_fill(**i),
        "flag_order_for_review": lambda i: flag_order_for_review(**i),
        "get_market_summary": lambda i: get_market_summary(),
    }
    handler = handlers.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"
    return handler(tool_input)

# ============================================================
# AGENT LOOP
# ============================================================

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a brokerage operations assistant for a financial trading firm.
You help operations managers analyze order flow, predict fill rates, and flag risky orders.

You have access to:
- Historical trade reports (via query_trade_reports)
- An ML model for fill prediction (via predict_order_fill)
- A compliance flagging system (via flag_order_for_review)
- Live market conditions (via get_market_summary)

Always be specific and data-driven. Reference actual numbers from tool results.
If an order has low fill probability (<40%) or high risk, proactively suggest flagging it."""

def run_agent(user_message: str):
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print('='*60)
    
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )
        
        # Check if we need to run tools
        if response.stop_reason == "tool_use":
            # Process all tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n🔧 Tool: {block.name}")
                    print(f"   Input: {json.dumps(block.input, indent=6)}")
                    result = execute_tool(block.name, block.input)
                    print(f"   Result: {result[:100]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add assistant response + tool results to history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
        else:
            # Final response
            final = "".join(b.text for b in response.content if hasattr(b, "text"))
            print(f"\nAGENT: {final}")
            return final

# Test scenarios
if __name__ == "__main__":
    scenarios = [
        "What's the current market condition and should I be cautious today?",
        
        "I have a stop order: 8000 units, Broker C, current volatility is 1.9, it's 14:00, "
        "and this client hasn't traded in 20 days. Should I be worried?",
        
        "Compare fill rates for market orders vs limit orders across all brokers "
        "and tell me which combination performs best.",
    ]
    
    for scenario in scenarios:
        run_agent(scenario)
        print()
```

- [ ] Run: `python src/brokerage_agent.py`
- [ ] Observe the tool call loop — agent reasons → calls tool → observes → reasons again
- [ ] **Key insight:** This is the ReAct loop. You built this in your workshop. Now it's domain-specific and uses real tools.

### 🧠 Add Conversation Memory (Day 4–5)
- [ ] Read: [Memory types in LangChain](https://python.langchain.com/docs/concepts/memory/) (~15 min)
- [ ] Add session memory to your agent:

```python
# Add to brokerage_agent.py

class BrokerageAgentSession:
    """Stateful agent with conversation memory"""
    
    def __init__(self):
        self.messages = []
        self.context = {}  # Persist facts across turns
    
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        
        # Build system with accumulated context
        context_str = "\n".join([f"- {k}: {v}" for k, v in self.context.items()])
        system = SYSTEM_PROMPT
        if context_str:
            system += f"\n\nContext from this session:\n{context_str}"
        
        # Run with full history
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            tools=tools,
            messages=self.messages
        )
        
        # Handle tool calls...
        # (same tool loop as before)
        
        final = "".join(b.text for b in response.content if hasattr(b, "text"))
        self.messages.append({"role": "assistant", "content": response.content})
        return final
```

- [ ] Test a multi-turn conversation:
  ```
  Turn 1: "What's the fill rate for Broker A?"
  Turn 2: "How does that compare to Broker B?"  ← agent remembers context
  Turn 3: "Given this, which broker should I route my next large order to?"
  ```

### 📝 Week 8 Wrap-up (Day 6–7)
- [ ] In `notes.md`, answer:
  1. What's the difference between your agent and a hardcoded if/else decision tree?
  2. When would you NOT use an agent here? (Critical thinking)
  3. How would you add this agent as a service in the Manafsoft architecture?
- [ ] Commit everything

---

## ✅ Week 9 Checklist

### 🔗 Connect Agent to Your Other Systems (Day 1–2)
- [ ] Make the agent call your actual RAG API and ML prediction API (not mocks)
- [ ] Run the full pipeline: question → RAG retrieval → ML prediction → agent synthesis
- [ ] Test with 5 realistic scenarios a brokerage ops manager might ask

### 📊 Add Observability (Day 3–4)
- [ ] Log every agent run: input, tools called, tool results, final output, latency
- [ ] Create `src/agent_logger.py`:

```python
import json
from datetime import datetime

class AgentLogger:
    def __init__(self, log_file="agent_runs.jsonl"):
        self.log_file = log_file
    
    def log_run(self, user_input, tools_called, final_output, latency_ms):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "tools_called": tools_called,
            "output_length": len(final_output),
            "latency_ms": latency_ms
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

- [ ] After 10 runs, read the log — what tools are called most? Where is latency highest?

### 🎯 BYOA Framework: Package Your Agent (Day 5–6)
- [ ] Create `src/agent_config.yaml`:

```yaml
# BYOA — Brokerage Operations Agent Configuration
# This is your methodology packaged as an agent

agent:
  name: "BrokerageOpsAgent"
  version: "1.0.0"
  domain: "financial-trading"
  
tools:
  - name: query_trade_reports
    endpoint: "http://rag-service:8001/ask"
    timeout_ms: 15000
  - name: predict_order_fill
    endpoint: "http://ml-service:8000/predict"
    timeout_ms: 5000
  - name: flag_order_for_review
    endpoint: "http://compliance-service:9000/flag"
    timeout_ms: 3000

behavior:
  max_iterations: 5
  auto_flag_threshold: 0.35  # flag orders with fill prob below this
  memory: session  # session | persistent | none
  
system_prompt: |
  You are a brokerage operations assistant...
```

- [ ] Write `BYOA_CONCEPT.md` — your thinking on what BYOA means, using THIS agent as an example. This becomes a LinkedIn article.

### 📝 Week 9 Wrap-up (Day 7)
- [ ] Full system test: all three services running (RAG, ML, Agent), run 5 end-to-end scenarios
- [ ] Document failures — what breaks, what would need to be hardened for production
- [ ] Commit all code + configs
- [ ] Update main README → Week 8–9 ✅ Done

---

## 📚 Resource Reference

| Resource | Type | Time | Priority |
|----------|------|------|----------|
| [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | Article | 45 min | 🔴 Must |
| [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) | Docs | 30 min | 🔴 Must |
| [ReAct Paper site](https://react-lm.github.io/) | Paper | 20 min | 🟡 Recommended |
| [LangChain Memory Concepts](https://python.langchain.com/docs/concepts/memory/) | Docs | 15 min | 🟡 Recommended |

---

## 🏁 End-of-Phase Self-Assessment

- [ ] I have a working multi-tool agent in a financial domain
- [ ] The agent uses ReAct pattern — reason, act, observe, reason again
- [ ] I've added session memory and observability
- [ ] I can articulate what BYOA means using my own agent as an example
- [ ] I know when NOT to use agents (important for credibility)

**Next:** [Week 10 — Public Visibility →](../week-09-visibility/README.md)
