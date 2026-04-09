# Week 3–4: Model Serving with FastAPI + Docker

> **Theme:** Take your trained model and turn it into a production API — the same way you'd wrap a Java service. This is where your existing engineering skills shine.
> **Time commitment:** ~1.5–2 hrs/day
> **Status:** ⬜ Not Started

---

## 🎬 Demoable Deliverable

**What to show:** A live REST API serving ML predictions from a Docker container — hit `/predict` with curl or Swagger UI and get real-time order fill predictions with confidence scores.

**Demo format:**
- Screen recording: `docker compose up` → MLflow UI on :5000 + Prediction API on :8000
- Live Swagger UI walkthrough — send requests, show responses with latency
- Show `/metrics` endpoint with Prometheus counters after multiple predictions

**Where to share:** GitHub repo with Docker instructions + a short Loom/screen recording. Deploy to a free tier (Railway, Render, or HuggingFace Spaces) for a live URL.

---

## 🎯 What You're Learning (and Why)

This is YOUR territory as a Java/Spring engineer. The concepts are identical — you're just using Python + FastAPI instead of Java + Spring Boot. By the end of this phase, you'll have a **running REST API that serves ML predictions**, containerized in Docker.

| Spring Boot Concept | FastAPI + MLflow Equivalent |
|--------------------|----------------------------|
| `@RestController` | `@app.post("/predict")` |
| `application.yml` | `config.py` or env vars |
| JAR packaging | Docker image |
| Spring Actuator `/health` | FastAPI `/health` endpoint |
| Dependency injection | MLflow model loading at startup |

---

## ✅ Week 3 Checklist

### 🔧 FastAPI Setup (Day 1)
- [ ] Install FastAPI and related packages:
  ```bash
  pip install fastapi uvicorn pydantic python-multipart
  ```
- [ ] Read: [FastAPI — First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/) (~20 min)
- [ ] Read: [FastAPI — Request Body (Pydantic models)](https://fastapi.tiangolo.com/tutorial/body/) (~15 min)
- [ ] **What to absorb:** How Pydantic models validate input — it's like Java Bean Validation (`@NotNull`, `@Size`) but cleaner. FastAPI auto-generates Swagger docs at `/docs`. That's it, you're ready.

### 🚀 Build Your First Model API (Day 2–3)

- [ ] Create `src/api.py` — a REST API serving your OrderFillPredictor:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Order Fill Predictor API",
    description="Predicts whether a brokerage order will be filled based on market conditions",
    version="1.0.0"
)

# Load model at startup (like @PostConstruct in Spring)
MODEL_URI = "models:/OrderFillPredictor/Staging"
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info(f"Model loaded from {MODEL_URI}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fall back to a local path for development
        model = mlflow.sklearn.load_model("./local_model")

# Input schema (like a Java DTO)
class OrderRequest(BaseModel):
    order_size: int = Field(..., gt=0, le=100000, description="Order size in units")
    broker_tier: str = Field(..., pattern="^[ABC]$", description="Broker tier: A, B, or C")
    market_volatility: float = Field(..., ge=0.1, le=5.0)
    time_of_day_hour: int = Field(..., ge=9, le=16)
    order_type: str = Field(..., pattern="^(market|limit|stop)$")
    days_since_last_trade: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "order_size": 3000,
                "broker_tier": "A",
                "market_volatility": 0.8,
                "time_of_day_hour": 10,
                "order_type": "market",
                "days_since_last_trade": 2
            }
        }

# Output schema
class PredictionResponse(BaseModel):
    order_fill_probability: float
    prediction: str
    confidence: str
    model_version: str = "OrderFillPredictor/Staging"
    latency_ms: float

# Encode categoricals (same mapping used in training)
BROKER_MAP = {"A": 0, "B": 1, "C": 2}
ORDER_TYPE_MAP = {"limit": 0, "market": 1, "stop": 2}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: OrderRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    features = np.array([[
        request.order_size,
        BROKER_MAP[request.broker_tier],
        request.market_volatility,
        request.time_of_day_hour,
        ORDER_TYPE_MAP[request.order_type],
        request.days_since_last_trade
    ]])
    
    proba = model.predict_proba(features)[0][1]  # probability of fill=1
    prediction = "WILL_FILL" if proba >= 0.5 else "WILL_NOT_FILL"
    confidence = "HIGH" if abs(proba - 0.5) > 0.3 else "MEDIUM" if abs(proba - 0.5) > 0.15 else "LOW"
    latency = (time.time() - start) * 1000
    
    logger.info(f"Prediction: {prediction} (prob={proba:.3f}, latency={latency:.1f}ms)")
    
    return PredictionResponse(
        order_fill_probability=round(float(proba), 4),
        prediction=prediction,
        confidence=confidence,
        latency_ms=round(latency, 2)
    )

@app.get("/health")
async def health():
    return {"status": "UP", "model_loaded": model is not None}

@app.get("/")
async def root():
    return {"service": "Order Fill Predictor", "docs": "/docs"}
```

- [ ] Save your trained model locally for dev:
  ```python
  # Run this once to save a local copy:
  import mlflow.sklearn
  model = mlflow.sklearn.load_model("models:/OrderFillPredictor/Staging")
  mlflow.sklearn.save_model(model, "./local_model")
  ```
- [ ] Start the API:
  ```bash
  uvicorn src.api:app --reload --port 8000
  ```
- [ ] Open http://localhost:8000/docs — you have Swagger UI!
- [ ] Test it: click "Try it out" on `/predict`, send a request, see the response
- [ ] **Key insight:** This is exactly what a model serving layer looks like. `/predict` endpoint, schema validation, health check. Same as your Spring Boot services.

### 🧪 Test Your API Properly (Day 4–5)
- [ ] Create `src/test_api.py`:
```python
import requests
import json

BASE_URL = "http://localhost:8000"

test_cases = [
    {
        "name": "High-confidence fill: small market order, low volatility",
        "payload": {"order_size": 1000, "broker_tier": "A", "market_volatility": 0.5,
                   "time_of_day_hour": 10, "order_type": "market", "days_since_last_trade": 1}
    },
    {
        "name": "Likely no-fill: huge stop order, high volatility",
        "payload": {"order_size": 9000, "broker_tier": "C", "market_volatility": 1.9,
                   "time_of_day_hour": 15, "order_type": "stop", "days_since_last_trade": 25}
    },
    {
        "name": "Edge case: mid-day limit order",
        "payload": {"order_size": 4500, "broker_tier": "B", "market_volatility": 1.0,
                   "time_of_day_hour": 12, "order_type": "limit", "days_since_last_trade": 5}
    }
]

print("Testing Order Fill Predictor API\n" + "="*40)
for tc in test_cases:
    r = requests.post(f"{BASE_URL}/predict", json=tc["payload"])
    result = r.json()
    print(f"\n{tc['name']}")
    print(f"  → {result['prediction']} ({result['confidence']} confidence)")
    print(f"  → Fill probability: {result['order_fill_probability']}")
    print(f"  → Latency: {result['latency_ms']}ms")

print("\n" + "="*40)
print(f"Health: {requests.get(f'{BASE_URL}/health').json()}")
```
- [ ] Run tests, see results
- [ ] Add one more test case of your own design

---

## ✅ Week 4 Checklist

### 🐳 Containerize with Docker (Day 1–2)
- [ ] Install Docker Desktop if not already installed
- [ ] Read: [Docker getting started](https://docs.docker.com/get-started/) — Part 1 & 2 only (~30 min)
- [ ] Create `Dockerfile` in your project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer caching — same as Maven)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY local_model/ ./local_model/

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Health check (like Kubernetes liveness probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] Create `requirements.txt`:
  ```
  fastapi==0.109.0
  uvicorn==0.27.0
  pydantic==2.5.0
  mlflow==2.10.0
  scikit-learn==1.4.0
  numpy==1.26.0
  pandas==2.2.0
  ```
- [ ] Build and run:
  ```bash
  docker build -t order-fill-predictor:v1 .
  docker run -p 8000:8000 order-fill-predictor:v1
  ```
- [ ] Test with `python src/test_api.py` — hits the containerized version
- [ ] **Key insight:** This Docker image is now a deployable artifact. Same as your JAR. You can push it to any container registry (ECR, GCR, Docker Hub) and run it anywhere.

### 📊 Add Logging + Prometheus Metrics (Day 3–4)

This is what separates "deployed" from "production-ready."

- [ ] Install prometheus client:
  ```bash
  pip install prometheus-client
  ```
- [ ] Add metrics to `src/api.py`:
```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response
import time

# Metrics (like Micrometer in Spring Boot)
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['result', 'confidence'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
REQUEST_ERRORS = Counter('prediction_errors_total', 'Total errors')

# Add to predict endpoint, inside the function:
PREDICTION_COUNT.labels(result=prediction, confidence=confidence).inc()
PREDICTION_LATENCY.observe(latency / 1000)

# Add a metrics endpoint:
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```
- [ ] Restart API, make 5–10 predictions, then hit `/metrics`
- [ ] You'll see raw Prometheus metrics — `predictions_total`, `prediction_latency_seconds`

### 🔗 Docker Compose: API + MLflow Together (Day 5)
- [ ] Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  mlflow:
    image: python:3.10-slim
    command: >
      bash -c "pip install mlflow && mlflow server 
      --host 0.0.0.0 --port 5000 
      --backend-store-uri sqlite:///mlflow.db 
      --default-artifact-root ./artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/app
    working_dir: /app

  predictor-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

volumes:
  mlflow-data:
```
- [ ] Run: `docker compose up`
- [ ] Both services start together — MLflow UI on :5000, API on :8000
- [ ] **Key insight:** This is a micro-service ML system. MLflow = model registry service, predictor-api = inference service. You've built a real architecture.

### 📝 Week 4 Wrap-up (Day 6–7)
- [ ] Create `src/README_API.md` documenting your API:
  - What it predicts
  - Input/output schema
  - How to run locally
  - How to run with Docker
  - Example curl commands
- [ ] Add a `ARCHITECTURE.md` in the week folder with a simple ASCII or mermaid diagram of the system
- [ ] Commit everything
- [ ] Update main README → Week 3–4 ✅ Done

---

## 📚 Resource Reference

| Resource | Type | Time | Priority |
|----------|------|------|----------|
| [FastAPI First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/) | Docs | 20 min | 🔴 Must |
| [FastAPI Request Body](https://fastapi.tiangolo.com/tutorial/body/) | Docs | 15 min | 🔴 Must |
| [Docker Getting Started](https://docs.docker.com/get-started/) | Docs | 30 min | 🔴 Must |
| [MLflow Model Serving](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models) | Docs | 20 min | 🟡 Recommended |
| [Prometheus Python Client](https://github.com/prometheus/client_python) | Docs | 20 min | 🟡 Recommended |
| [Docker Compose](https://docs.docker.com/compose/gettingstarted/) | Docs | 20 min | 🟡 Recommended |

---

## 🏁 End-of-Phase Self-Assessment

- [ ] I have a running REST API that serves ML model predictions
- [ ] My API has input validation, health check, and metrics endpoint
- [ ] My API runs in Docker
- [ ] I can explain how this maps to my existing Spring Boot service knowledge
- [ ] I have Docker Compose running MLflow + API together

**Next:** [Week 5–6 — RAG Pipeline on Financial Data →](../week-05-06-rag-pipeline/README.md)
