# LinkedIn Post 2: Data Drift Detection

---

## Version A — "Here's what I built"

AI Engineering pivot — Week 1, Part 2.

Last post I tracked 6 ML experiments with MLflow. This time: what happens AFTER you deploy a model.

Spoiler: it starts decaying immediately.

What I did:
→ Trained an Order Fill Predictor on brokerage data from last quarter
→ Deployed it — predictions looked great
→ Simulated real-world conditions: market volatility shifts, order sizes change
→ Ran drift detection using Evidently AI on the model's input features
→ Result: 2 out of 3 features had drifted significantly

Without this, you're guessing when to retrain. With it, you have evidence.

Deploying a model is half the job. Knowing when it's gone stale is the other half.

Next up: wrapping this model in a FastAPI endpoint and deploying it in Docker. The journey continues.

Have you seen models silently degrade in production? How did your team catch it?

#AIEngineering #MLOps #DataDrift #MachineLearning #BuildingInPublic

---

## Version B — "Value-add insight"

AI Engineering pivot — Week 1, Part 2.

Your ML model is getting worse right now. You just don't know it yet.

Models are trained on historical data. But the world doesn't stand still.

In financial markets this is obvious:
→ Volatility spikes
→ Order sizes shift
→ New trading patterns emerge

A model trained on last quarter's data may not reflect this quarter's reality. This is called data drift — when the real-world data moves away from what the model was trained on.

The fix isn't retraining on a schedule. It's monitoring your model's inputs and retraining when the data tells you to.

I set this up using Evidently AI on brokerage order data. The drift report showed 2 of 3 input features had shifted significantly. That's not a "maybe retrain" — that's a clear signal.

Monitoring model inputs is just as important as monitoring model outputs. The tooling exists. Most teams just aren't using it yet.

Following the journey? Next: deploying this model as a live API in Docker.

#AIEngineering #MLOps #DataDrift #MachineLearning #BuildingInPublic
