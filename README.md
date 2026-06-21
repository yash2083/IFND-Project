# IFND: Indian Fake News Detection 🇮🇳📰

Welcome to the **Indian Fake News Detection (IFND)** project repository! This project focuses on identifying and classifying fake news articles, statements, and claims within the highly diverse and multilingual Indian context.

By leveraging advanced Natural Language Processing (NLP) techniques and state-of-the-art Transformer models (like **MuRIL** and **IndicBERT**), this pipeline aims to combat misinformation with high accuracy and explainability.

---

## 📊 Dataset Overview
The dataset comprises a total of **56,714 statements**, curated to reflect news and claims circulating in India.

* **Target Labels**: 
  * `True` (Real News): ~66.6% (37,800 records)
  * `Fake` (Fake News): ~33.3% (18,914 records)
* **Categories Covered**: Government, Violence, COVID-19, Politics, Elections, Terror, and Misleading content.
* **Data Splits**: 
  * Training Set: 39,721 
  * Validation Set: 8,485 
  * Test Set: 8,508

---

## ⚙️ Entire Machine Learning Pipeline

The project follows a rigorous, end-to-end Machine Learning pipeline entirely documented in the `IFND_EDA.ipynb` notebook.

### 1. Data Preprocessing & EDA (Exploratory Data Analysis)
* **Null Value Handling**: Identified and addressed missing data (e.g., missing dates).
* **Text Cleaning**: Applied standard NLP cleaning procedures (removing special characters, standardizing text, handling code-mixed nuances).
* **Class Imbalance Review**: Evaluated the True vs. Fake label distribution to ensure robust train/validation splitting.

### 2. Baseline Modeling (TF-IDF + XGBoost)
Before moving to deep learning, a strong machine learning baseline was established.
* **Feature Engineering**: Extracted text features using TF-IDF (Term Frequency-Inverse Document Frequency).
* **Model**: Trained an XGBoost classifier.
* **Baseline Result**: Achieved a highly respectable **87.22% Accuracy**.

### 3. Advanced Transformer Fine-Tuning
To capture deep contextual semantics, specifically for Indian geopolitics and code-mixed languages, several Transformer architectures were fine-tuned:
* **mBERT (Multilingual BERT)**: 98.00% Accuracy
* **IndicBERTv2 (AI4Bharat)**: 98.00% Accuracy
* **MuRIL (Multilingual Representations for Indian Languages)**: **97.61% Accuracy**

**🏆 Proposed Model: MuRIL**
While mBERT and IndicBERT performed incredibly well, **MuRIL** was selected as the proposed model. Developed specifically by Google for Indian languages, MuRIL provides superior out-of-the-box support for transliterated and code-mixed text (e.g., Hinglish), which is essential for scaling real-world Indian fake news detection.

### 4. Explainable AI (SHAP Analysis)
Detecting fake news is only half the battle; explaining *why* it is fake builds trust. We utilized **SHAP (SHapley Additive exPlanations)** to interpret the MuRIL model's black-box predictions.

**Key Findings:**
* **Fake News Triggers:** Words like `"Fact"`, `"Check"`, `"Fake"`, `"False"`, and `"viral"` were the strongest indicators of the Fake class, primarily because the dataset includes debunked claims and fact-checking reports.
* **Real News Triggers:** Words denoting attribution and facts like `"tells"`, `"says"`, `"Bharat"`, `"Ladakh"`, and `"Delhi"`.

---

## 📁 Repository Structure

```text
├── IFND_EDA.ipynb       # Core Jupyter Notebook with the entire EDA, Training, and Evaluation pipeline
├── results/             # Directory containing all output assets
│   ├── findings.md              # Summary of the model metrics and SHAP explainability
│   ├── baseline_results.csv     # Baseline XGBoost predictions/metrics
│   ├── final_results.csv        # Transformer evaluation metrics
│   ├── ifnd_train.csv           # Training dataset split
│   ├── ifnd_val.csv             # Validation dataset split
│   ├── ifnd_test.csv            # Test dataset split
│   ├── shap_top20_fake.png      # SHAP visualization for Fake News tokens
│   └── shap_top20_true.png      # SHAP visualization for Real News tokens
└── LICENSE              # MIT License
```

---

## 🚀 Getting Started

To explore this project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yash2083/IFND-Project.git
   cd IFND-Project
   ```

2. **Set up the Environment:**
   Install Jupyter and the necessary data science libraries (PyTorch, Transformers, XGBoost, SHAP, Pandas).
   ```bash
   pip install torch transformers xgboost shap pandas scikit-learn jupyter
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook IFND_EDA.ipynb
   ```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
