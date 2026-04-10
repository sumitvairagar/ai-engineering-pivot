# LinkedIn Post 1: MLflow Experiment Tracking

---

## Version A — "Here's what I built"

I'm pivoting into AI Engineering. Here's Week 1.

After 14 years building Java/Spring microservices and brokerage systems, I'm adding AI to my toolkit — and documenting the whole journey.

Week 1: Experiment Tracking with MLflow.

What I did:
→ Built an Order Fill Predictor on brokerage data using scikit-learn
→ Ran 6 experiment runs with different hyperparameters
→ Tracked every run in MLflow — accuracy, F1 score, learning rate, tree depth
→ Compared all 6 side-by-side in one dashboard
→ Promoted the best model to Production with a single alias change

The insight: We version code with Git. We track deployments with CI/CD. ML experiments deserve the same discipline. MLflow brings that — open-source, works with any framework.

This is the foundation. Next up: deploying this model as a live API.

What's your experience with ML experiment tracking? Curious if teams are actually using it or still eyeballing Jupyter notebooks.

#AIEngineering #MLOps #MLflow #MachineLearning #BuildingInPublic

---

## Version B — "Value-add insight"

I'm pivoting into AI Engineering after 14 years in Java/Spring/Kafka. Documenting the journey — here's Week 1.

Most ML models die in notebooks. Here's why — and what production teams do differently.

The gap between "I trained a model" and "this model is in production" is massive. It's not a code problem. It's a process problem.

In software, we solved this years ago:
→ Git for code versioning
→ CI/CD for deployments
→ Monitoring for production health

ML needs the same:
→ Experiment tracking (what did I try? what worked?)
→ Model registry (which version is live?)
→ One-click promotion and rollback

I set this up on brokerage order data using MLflow. 6 experiment runs, all tracked, compared side-by-side, best one promoted to production.

The tooling is mature. The patterns are borrowed from software engineering. If your team is still picking models by scrolling through Jupyter notebooks — there's a better way.

Following along? Next week I'm deploying this model as a live API.

#AIEngineering #MLOps #MLflow #MachineLearning #BuildingInPublic
