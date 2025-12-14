## Credit Scoring Business Understanding

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