# Phases 19-22: Deployment, Evaluation, and Wrap

## Phase 19: Public deployment

The repository contains the lightweight FastAPI/UI core suitable for a free host. Actual public deployment requires a hosting account and user-controlled credentials, so no public URL is claimed here.

## Phase 20: Evaluation

The current smoke tests verify code paths only. They do not constitute parsing precision/recall, ranking Precision@K, Recall@K, NDCG, or semantic-quality metrics. Those require manually labeled evaluation data and executable evaluation scripts.

## Phase 21: MLOps

The design keeps source provenance, schema version, warnings, and component scores. Future monitoring should track extraction failure rates, score distributions, drift, model versions, and retraining data quality. Recruiter accept/reject signals must not be used blindly as labels because they may encode selection bias.

## Phase 22: Final wrap

Implemented and deferred scope is documented in this project and phase files. A final slide deck and public link must use real screenshots and measured metrics after the local stack and evaluation corpus exist.
