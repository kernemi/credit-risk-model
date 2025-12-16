## Credit Scoring Business Understanding

## Task1-Understanding Credit Risk
### Overview

Credit risk represents the possibility that a borrower will fail to meet their contractual repayment obligations, leading to financial loss for the lender. In the context of this project, credit risk analysis is essential for evaluating customer creditworthiness and supporting responsible lending decisions, particularly in environments such as Buy Now Pay Later (BNPL) and alternative credit scoring where traditional credit histories may be limited or unavailable.

Effective credit scoring models enable financial institutions to quantify risk, price credit appropriately, manage portfolios, and comply with regulatory requirements. This project aligns with international best practices, particularly the Basel II Capital Accord, which emphasizes robust, transparent, and well-documented risk measurement frameworks.

### How Basel II Influences the Need for Interpretable and Well-Documented Models

The Basel II Capital Accord places strong emphasis on risk-sensitive capital requirements, encouraging banks to use internal models to estimate credit risk components such as Probability of Default (PD). However, Basel II also requires that these models be:
```
Transparent and explainable

Well-documented

Auditable and justifiable to regulators
```

As a result, credit risk models must not only be accurate but also interpretable. Regulators and internal risk committees need to understand why a borrower is classified as high or low risk. Models such as Logistic Regression with Weight of Evidence (WoE) transformations are widely accepted because they provide clear relationships between input variables and predicted risk. This interpretability ensures regulatory compliance, supports governance, and builds trust in automated credit decisions.

### Use of a Proxy Default Variable and Associated Business Risks

In many real-world datasets especially in alternative lending contexts there is no explicit or labeled “default” variable. Therefore, it becomes necessary to construct a proxy default variable, such as defining default based on missed payments, delinquency thresholds, or behavioral indicators.

Creating a proxy variable is necessary because:

    -Supervised learning models require a target variable.

    - True defaults may be rare, delayed, or not formally recorded.

    - BNPL and alternative credit products often lack long-term default histories.

However, using a proxy introduces business risks:

    - Label noise: The proxy may not perfectly represent true default behavior.

    - Model bias: Incorrect assumptions may lead to unfair or inaccurate predictions.

    - Mispricing risk: Overestimating or underestimating risk can reduce profitability.

    - Strategic risk: Decisions based on imperfect labels may increase rejection of good customers or approval of risky ones.

To mitigate these risks, proxy definitions must be carefully designed, validated with domain experts, and continuously monitored.

### Trade-Offs Between Interpretable and High-Performance Models

In regulated financial environments, there is a critical trade-off between model interpretability and predictive performance.

Simple, Interpretable Models (e.g., Logistic Regression with WoE):

    ✔ Easy to explain to regulators and stakeholders

    ✔ Transparent variable contributions

    ✔ Easier governance and compliance

    ✖ Limited ability to capture complex non-linear relationships

    ✖ May have lower predictive accuracy

Complex Models (e.g., Gradient Boosting Machines):

    ✔ Higher predictive accuracy

    ✔ Better handling of non-linear interactions

    ✔ Strong performance on large, complex datasets

    ✖ Limited interpretability (“black-box” nature)

    ✖ Increased regulatory scrutiny and explainability challenges

    ✖ Higher operational and monitoring complexity

In practice, financial institutions often adopt a balanced approach, using interpretable models for regulatory reporting and decision justification, while leveraging more complex models in controlled environments where explainability tools (XAI) and governance frameworks are in place.

### Conclusion

Understanding credit risk from both a business and regulatory perspective is foundational to building effective credit scoring systems. Basel II reinforces the need for accurate, interpretable, and well-governed models. While alternative data and advanced machine learning techniques offer improved predictive power, responsible credit risk modeling requires careful consideration of transparency, proxy target construction, and regulatory constraints. This project reflects these principles by prioritizing explainability, robustness, and business relevance in credit risk modeling.

## Task2- Exploratory Data Analysis (EDA)
- The step by step procedure and findings is on the 
```
notebooks/eda.ipynb
```
### Key EDA Insights
1. Transaction amounts are heavily right-skewed.Most transactions have very small amounts.A small number of transactions have extremely large values.
2. Majority of users transact infrequently.Very few customers are frequent users.
3. Fraudulent transactions are rare or nearly zero.
4. A small number of product categories dominate volume.
5. Significant variance across customers.Some spend very little, others spend a lot.

## Task3– Feature Engineering Pipeline
File: 
```
src/data_processing.py
```
Steps:
1. Aggregate per customer: total_amount, avg_amount, txn_count, std_amount
```
df_grouped = df.groupby("CustomerId").agg(
    total_amount=("Amount", "sum"),
    avg_amount=("Amount", "mean"),
    txn_count=("TransactionId", "count"),
    std_amount=("Amount", "std")
).reset_index()
```
2. Date-time features: Extract hour, day, month, year from TransactionStartTime.
3. Categorical encoding: OneHotEncoder for ProductCategory, ChannelId, ProviderId, PricingStrategy.
4. Handle missing values: SimpleImputer (median/mean for numerics, mode for categoricals).
5. Scaling:StandardScaler for numeric features.
6. Pipeline: Use ColumnTransformer + Pipeline to combine transformations.
7. Weight of Evidence (WoE): Apply for Logistic Regression variant using xverse.WOE.
Output: data/processed/features.csv.

## Task 4 – Proxy Target Variable (RFM + Clustering)

Steps:
- Compute RFM metrics per customer:
```
Recency: Days since last transaction
Frequency: Number of transactions
Monetary: Total transaction amount
```
- Scale features → KMeans clustering (3 clusters)
- Identify high-risk cluster: High recency, low frequency, low monetary
- Create binary target is_high_risk
- Merge target with processed features
- Output: final_df ready for model training

## Task 5 – Model Training & MLflow

Steps:

-Split X/y → train/test
-Train at least 2 models: Logistic Regression, Random Forest/Gradient Boosting
-Hyperparameter tuning (GridSearchCV)
-Track experiments with MLflow: parameters, metrics, model artifact
-Register best model in MLflow
-Unit tests: check feature columns and is_high_risk column
-Output: Trained model saved (best_model.pkl) and logged in MLflow

## Task 6 – API, Docker & CI/CD

FastAPI (src/api/main.py):
```
/ → health check
/predict → returns risk_probability using trained model
Input validated via Pydantic (InputData)
```
Docker:
```
Dockerfile → builds API with Python dependencies

docker-compose.yml → runs container on port 8000
```
GitHub Actions CI/CD:
```
Installs dependencies

Runs flake8 for linting

Runs pytest for unit tests

Fails build if lint/tests fail
```

## Optional Enhancements
- MLflow model versioning to auto-load latest registered model in API
-Logging & error handling in FastAPI endpoints
- Extend CI/CD to build Docker image and push to registry (for production deployment)