-- ==========================================================
-- Telco Customer Churn - SQL Analytics & Segmentation
-- ==========================================================

-- 1. Overall Churn Rate & Customer Count
SELECT 
    COUNT(*) AS total_customers,
    SUM(ChurnFlag) AS churned_customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct
FROM telco_churn;

-- 2. Churn Rate by Contract Type
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(ChurnFlag) AS churned_customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn by Internet Service & Tech Support
SELECT 
    InternetService,
    TechSupport,
    COUNT(*) AS total_customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY InternetService, TechSupport
ORDER BY InternetService, churn_rate_pct DESC;

-- 4. Churn Risk by Tenure Buckets
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-1 Year'
        WHEN tenure <= 24 THEN '1-2 Years'
        WHEN tenure <= 36 THEN '2-3 Years'
        WHEN tenure <= 48 THEN '3-4 Years'
        WHEN tenure <= 60 THEN '4-5 Years'
        ELSE '5+ Years'
    END AS tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(ChurnFlag) AS churned_customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct
FROM telco_churn
GROUP BY 1
ORDER BY churn_rate_pct DESC;

-- 5. Payment Method & Paperless Billing Impact
SELECT 
    PaymentMethod,
    PaperlessBilling,
    COUNT(*) AS total_customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct
FROM telco_churn
GROUP BY PaymentMethod, PaperlessBilling
ORDER BY churn_rate_pct DESC;

-- 6. High-Value At-Risk Customers (Priority for Retention)
SELECT 
    customerID,
    tenure,
    Contract,
    InternetService,
    MonthlyCharges,
    TotalCharges
FROM telco_churn
WHERE ChurnFlag = 0
  AND Contract = 'Month-to-month'
  AND MonthlyCharges > 70
  AND tenure <= 12
ORDER BY MonthlyCharges DESC;
