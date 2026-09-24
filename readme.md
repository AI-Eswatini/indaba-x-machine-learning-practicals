# IndabaX Resilient Credit ML Practical

*Introduction to Google Colab & Kaggle, then a full resilient credit-decision pipeline built for IndabaX Eswatini, under the theme "Ensemble Machine Learning for Resilient Decision Making in Eswatini."*

---

## Why this notebook exists

Most ML tutorials show you how to train a model and report its accuracy. This one asks a harder question at every single step: **does this technique actually earn its place, or are we just assuming the fancier option is better?**

Every method in this pipeline was tested against a simpler alternative before being kept:
- Does GAIN actually beat median imputation, or just look more sophisticated?
- Does averaging two models actually beat the better one alone?
- Does scaling the data actually matter for a tree-based model?


---

## Before the pipeline: Colab & Kaggle

The first part of this session is a hands-on introduction for anyone new to the tools:
- **Google Colab** :running Python in the browser.
- **Kaggle** :finding, evaluating, and downloading real datasets, including how to spot a dataset that looks good but has no real signal.
---

## The pipeline, at a glance

| Stage | Method | Why |
|---|---|---|
| Imputation | GAIN (built from scratch) | Preserves realistic variance in filled-in values, where simple methods flatten it |
| Scaling | RobustScaler → Yeo-Johnson | Tames heavy-tailed income/wealth fields |
| Feature selection | Boruta-SHAP (built from scratch) | Statistically tests each feature against a random "shadow" version of itself |
| Prediction | Monotonic XGBoost vs. TabNet | Two structurally different models, benchmarked honestly against each other |
| Explainability | SHAP + DiCE | Answers both "why was I rejected" and "what would change it" |

---

## Getting started

**Fastest path — Google Colab:**

1. Open `Introduction_to_Google_Colab_&_Kaggle.ipynb` in Colab.
2. Run the setup cell (installs `pytorch-tabnet`, `dice-ml`, `shap`, `xgboost`, everything else is already in Colab).
3. Upload `Kaggle_Loan_Data.csv` when prompted.
4. Run every cell, top to bottom, in order.

> **Edited an earlier cell?** Use `Runtime > Restart session`, then `Runtime > Run all`, before trusting any result. This notebook has several cells that depend on things defined earlier, skipping this step is the most common source of confusing errors.

**Running locally instead:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch pytorch-tabnet shap dice-ml jupyter
jupyter notebook Introduction_to_Google_Colab_&_Kaggle.ipynb
```

`Kaggle_Loan_Data.csv` is included in this repository.

---

## After the session

- Run the same pipeline on a different Kaggle credit dataset and compare.
- Try Kaggle Learn's free micro-courses on feature engineering and model explainability.
- Look for African-context datasets and competitions on [Zindi](https://zindi.africa) to keep practicing.


