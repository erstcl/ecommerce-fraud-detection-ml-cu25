[🇷🇺 Русская версия](README.ru.md) | [🇬🇧 English version](README.md)

# E-Commerce Fraud Detection with SHAP Interpretability

**Final project**: Machine Learning Course (Central University)  
**Timeline**: November — December 2024

***

## Challenge

Build a machine learning system to detect fraudulent e-commerce transactions based on user behavioral patterns, transaction data, and security parameters. Fraud detection is mission-critical for online business: missed fraud = direct losses, false positives = lost customers.

***

## Dataset

**Source**: [E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset) (Kaggle)

### Characteristics
- **Size**: 299,695 transactions
- **Features**: 17 original features
- **Target variable**: `is_fraud` (binary classification)
- **Class imbalance**: 2.206% fraud (6,612 out of 299,695)
- **Time period**: 2024

### Feature Groups

**User profile**:
- `account_age_days` — account age
- `total_transactions_user` — user transaction count
- `avg_amount_user` — user average transaction amount

**Transaction**:
- `amount` — transaction amount
- `shipping_distance_km` — shipping distance
- `promo_used` — promo code usage
- `merchant_category` — merchant category
- `channel` — transaction channel (web/app)

**Security**:
- `avs_match` — Address Verification System match
- `cvv_result` — CVV verification result
- `three_ds_flag` — 3D Secure flag

**Geo**:
- `country` — user country
- `bin_country` — card issuer country

**Time**:
- `transaction_time` — transaction timestamp

***

## Solution

Project completed in 3 stages following university course roadmap:

### Stage 1: EDA & Baseline

**Data insights**:
- Fraud rate significantly higher for cross-border transactions (11.28% vs 1.43%)
- Top-2 fraud countries: TR (2.80%), RO (2.40%)
- Low fraud rate for transactions with all security checks (0.58% for AVS+CVV+3DS)
- Outliers: 11 extreme values in `amount` and `shipping_distance_km`

**Baseline**:
- Model: CatBoostClassifier
- Validation strategy: Stratified 80/20 split
- Metrics: ROC-AUC = 0.97784, PR-AUC = 0.85269

### Stage 2: Anomaly Detection & Feature Engineering

**Anomaly detection**:

3 approaches applied:

1. **Statistical methods** (Z-score, IQR):
   - `amount`: Z-score outliers (1.79% of data) → fraud rate 30.16% (14× base rate)
   - `shipping_distance_km`: Z-score outliers (3.61%) → fraud rate 17.27% (8× higher)
   - IQR outliers for `shipping_distance_km` → fraud rate 12.81%

2. **Extreme outlier removal**:
   - Removed points with `amount > 10000` and `shipping_distance_km > 10000`
   - Result: improved model stability

3. **ML methods for complex anomaly detection**:
   - Applied: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
   - Created `anomaly_count` (consensus anomaly counter)
   - Created `consensus_strong_anomaly` (points flagged by ≥2 methods)
   - 11,808 points identified as strong anomalies → fraud rate 23.76% (11× higher!)
   - **Insight**: Isolation Forest and LOF showed best precision/recall trade-off

**Feature Engineering**:

4 groups of new features created:

1. **Target Encoding** (leak-free):
   - `merchant_category_te`, `country_te`, `bin_country_te`
   - `channel` encoded via One-Hot Encoding

2. **User behavior features**:
   - `amount_zscore_user` — Z-score of amount relative to user average
   - `dist_zscore_user` — Z-score of shipping distance
   - `merchant_category` × `amount` — cross-features
   - kNN-based: transaction density in feature space

3. **Temporal features** (sin/cos encoding):
   - `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`
   - `is_night`, `is_business_hours`, `is_evening`, `is_weekend`

4. **Domain-specific features**:
   - `is_cross_border` — country ≠ bin_country (massive fraud indicator!)
   - `security_score` — AVS, CVV, 3DS combination (weighted score)
   - `all_security_passed` / `no_security` — flags
   - `amount_to_avg_ratio`, `amount_diff_from_avg` — deviations from user baseline
   - `is_long_distance` — long shipping distance flag (90th percentile)
   - **`risk_score`** — composite score:  
     `risk_score = 3×is_cross_border + 2×no_security + 1×three_ds_flag + is_long_distance + is_night`

**Feature selection**:
- Applied CatBoost feature importances
- Selected top-25 features for final model
- Removed unstable and duplicate features

**Result**: Baseline improved after feature engineering (details in notebook)

### Stage 3: Interpretability & Shapley Flow

**Model interpretation**:

1. **SHAP global interpretation**:
   - Built SHAP summary plots for CatBoost
   - Top influential features: `security_score`, `risk_score`, `amount`, `cross_border`, `shipping_distance`

2. **LIME local interpretation**:
   - Local interpretation of fraud transactions
   - LIME vs SHAP comparison: LIME shows simpler linear approximations, SHAP provides full interaction picture

3. **Model comparison**:
   - Compared Logistic Regression (with StandardScaler) and CatBoost
   - SHAP summary plot shows CatBoost better captures non-linear patterns (e.g., U-shaped dependencies)

**SHAP embeddings and anomalies**:

1. **SHAP embedding creation**:
   - Function `get_shap_embeddings(model, X_data, shap_feature)` for extracting SHAP values
   - SHAP embeddings for train and test

2. **Anomaly detection on SHAP embeddings**:
   - Isolation Forest with `contamination=0.01` on SHAP space
   - Identified 2,398 SHAP anomalies
   - **Result**: ROC-AUC = 0.97340 (slight decrease), but model became more stable

3. **SHAP embedding clustering**:
   - **PCA** for dimensionality reduction to 2 components
   - **k-Means** (k=5) → added `cluster` feature
   - Retrained CatBoost with `cat_features=['cluster']`
   - **Result**: ROC-AUC = 0.97566 (small improvement)
   - **DBSCAN**: no significant improvement (many outliers in cluster=-1)

**Shapley Flow analysis**:

1. **Feature interaction graph**:
   - Built graph based on SHAP value correlations (|corr| > 0.5)
   - NetworkX for visualization
   - Community detection (greedy modularity) → identified **18 feature groups**

2. **Key communities**:
   - Security cluster: `shap_security_score`, `shap_avs_match`, `shap_risk_score`, `shap_all_security_passed`
   - Geography cluster: `shap_shipping_distance_km`, `shap_is_cross_border`, `shap_is_long_distance`
   - User behavior: `shap_user_amount_std`, `shap_avg_amount_user`, `shap_anomaly_consensus`

3. **Train vs Test comparison**:
   - Test graph more sparse (5 communities vs 4 in train)
   - 4 stable groups persist between train/test
   - **Insight**: `shap_is_cross_border` and `shap_shipping_distance_km` always cluster together → strong relationship

**Final validation**:

Comparison of 3 approaches:
1. **SHAP embeddings + Isolation Forest**: ROC-AUC = 0.97340
2. **SHAP embeddings + clustering**: ROC-AUC = 0.97566
3. **Original features (hold-out validation)**: ROC-AUC = **0.97640**

SHAP embeddings alone (without original features): ROC-AUC = 0.96381 (1.26pp lower)

***

## Results

### Metrics (final model, hold-out validation)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.97640** |
| **PR-AUC (Average Precision)** | **0.85556** |
| **Fraud Precision** | 0.34 |
| **Fraud Recall** | 0.90 |
| **Fraud F1-Score** | 0.49 |

**Improvement over baseline**:
- ROC-AUC: +0.00144 (+0.15%)
- PR-AUC: +0.00287 (+0.34%)
- Fraud Precision: +0.04 (+13.3%)
- Fraud Recall: +0.02 (+2.3%)
- Fraud F1: +0.04 (+8.9%)

### Key Insights

**Business findings**:
1. **Cross-border transactions** — primary fraud indicator (fraud rate 11.28% vs 1.43%)
2. **Security checks critical**: AVS+CVV+3DS combination reduces fraud by 32× (0.58% vs 18.08%)
3. **Amount and distance anomalies** — strong signals (fraud rate up to 30% for outliers)
4. **Geography matters**: TR and RO — top fraud countries

**Technical findings**:
1. Feature engineering has greater impact than hyperparameter tuning
2. SHAP embeddings useful for interpretation but don't replace original features
3. Anomaly detection methods help identify complex patterns (consensus approach effective)
4. Shapley Flow reveals feature interaction structure

***

## Tech Stack

### ML Stack

**Data processing**:
- pandas, numpy
- scikit-learn (preprocessing, imputation, feature selection)

**Visualization**:
- matplotlib, seaborn
- plotly (interactive graphs)

**Anomaly detection**:
- Isolation Forest, LOF, One-Class SVM, Elliptic Envelope

**Modeling**:
- CatBoost (main model)
- Logistic Regression (for comparison)
- RandomForest (feature importance)

**Interpretability**:
- SHAP (TreeExplainer, summary plots, dependence plots)
- LIME (Tabular explainer, SP-LIME)

**Graph analysis**:
- NetworkX (Shapley Flow, community detection)

**Clustering**:
- k-Means, DBSCAN
- PCA, UMAP (dimensionality reduction)

### Environment

- **Platform**: Google Colab
- **Language**: Python 3.12
- **GPU**: NVIDIA Tesla T4 (for CatBoost acceleration)

***

## About the Project

Project completed as final team assignment for "Machine Learning" course at Central University. Project structure follows 3-stage roadmap:

1. **Checkpoint 1** (Nov 17-26): EDA + Baseline
2. **Checkpoint 2** (Nov 24 — Dec 3): Anomaly Detection + Feature Engineering
3. **Checkpoint 3** (Dec 1-10): SHAP Interpretability + Shapley Flow

Team completed all checkpoints, applying both classical statistical methods and SHAP embeddings with graph-based feature analysis.
