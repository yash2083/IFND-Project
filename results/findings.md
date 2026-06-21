# IFND Project Findings and Results

## 1. Dataset Overview
- **Total Records:** 56,714 statements
- **Columns:** `id`, `Statement`, `Image`, `Web`, `Category`, `Date`, `Label`
- **Label Distribution:** 
  - True: 37,800 (~66.6%)
  - Fake: 18,914 (~33.3%)
- **Data Splits:** Train: 39,721 | Val: 8,485 | Test: 8,508

## 2. Model Performance Comparison

| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
| --- | --- | --- | --- |
| TF-IDF + XGBoost | 87.22% | 86.63% | 84.44% |
| mBERT | 98.00% | 98.00% | 97.00% |
| IndicBERTv2 | 98.00% | 98.00% | 98.00% |
| **MuRIL (Proposed)** | **97.61%** | **97.61%** | **97.30%** |

*Note: While mBERT and IndicBERTv2 achieved slightly higher accuracy on the test set, MuRIL was chosen as the proposed model due to its specific pre-training on Indian languages and code-mixed text, making it highly robust for real-world Indian Fake News Detection (IFND).*

## 3. Explainability (SHAP Analysis)

We utilized SHAP (SHapley Additive exPlanations) to interpret the model's predictions and identify which words contributed most heavily to the classification of Fake vs. Real news.

### Top Tokens Driving "Fake News" Classification:
1. `Fact` (Importance: 15.79)
2. `Did` (Importance: 2.85)
3. `Fake` (Importance: 2.64)
4. `shared` (Importance: 2.46)
5. `Check` (Importance: 1.78)
6. `Old`, `video`, `False`, `viral`

*Observation: Fake news statements heavily feature words associated with fact-checking terminology ("Fact Check", "Fake", "False") or virality ("shared", "viral", "video"), often because the dataset contains statements debunking viral claims.*

### Top Tokens Driving "Real News" Classification:
1. `tells` (Importance: 0.42)
2. `says` (Importance: 0.35)
3. `Bharat` (Importance: 0.29)
4. `propaganda` (Importance: 0.28)
5. `claiming` (Importance: 0.25)
6. `Ladakh`, `Article`, `Delhi`

*Observation: Real news relies heavily on attribution ("tells", "says", "claiming") and specific entities/locations ("Bharat", "Ladakh", "Delhi").*
