# GNN-Based Cyber Attack Detection and Prevention in IoT Networks

A research project using CIC-IoT-2023 network traffic to compare classical ML models with Graph Neural Networks (GraphSAGE and GAT), followed by inference, risk assessment, and simulation-only prevention.

## Development order
1. Data ingestion and validation
2. Cleaning: NaN, Inf, duplicates
3. Feature preprocessing
4. Train/validation/test split
5. Class imbalance handling
6. Real network-relationship graph construction
7. Logistic Regression, XGBoost, MLP
8. GraphSAGE and GAT
9. Hyperparameter tuning
10. Evaluation and saved artifacts
11. New-traffic inference
12. Risk assessment and simulated response
13. Streamlit dashboard
14. Testing and documentation

Do not place the full dataset in Git. Put raw CIC-IoT-2023 files under `data/raw/`.
