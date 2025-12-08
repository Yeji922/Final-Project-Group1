````markdown
#  Suicide Ideation Detection — Improved Baseline Model  
Classical NLP + Logistic Regression + TF-IDF

This repository contains an **improved baseline model** for detecting suicide ideation from Reddit posts.  
The task is to classify posts into:

- **suicide**
- **non-suicide**

This model is part of a team project where my specific responsibility was:

> **Improve the baseline classical machine learning model to achieve strong, reliable performance.**

I redesigned and strengthened the baseline using advanced text preprocessing, expanded TF-IDF features, hyperparameter tuning, and class imbalance handling.

---

#  1. Overview  
The dataset contains posts from mental health–related subreddits:

- **r/SuicideWatch** → suicidal ideation  
- **r/depression** → mental health  
- **r/teenagers** → normal posts (non-suicide)  

The data was collected using the Pushshift API and is used in the research paper:

> *“Suicide Ideation Detection in Social Media Forums”*  
> https://ieeexplore.ieee.org/document/9591887

This dataset is widely used for mental-health NLP research.

---

#  2. Dataset Summary
- **Total posts:** ~232,000  
- **Labels:**  
  - suicide  
  - non-suicide  
- **Sources:** Reddit posts (2008–2021)  
- **Provided features:** title, post_text  

During preprocessing, I merged `title` + `post_text` to create the final text input.

---

#  3. Original Baseline Model  
The initial baseline used:

- basic text cleaning  
- TF-IDF with 1–2 grams  
- Logistic Regression with default settings  
- limited feature space  
- no hyperparameter tuning  
- minimal handling of class imbalance  

This left significant room for improvement.

---

#  4. Improved Baseline Model (My Contribution)
I upgraded the classical ML baseline in **three major ways**, keeping it simple, interpretable, and efficient.

---

##  A. Enhanced TF-IDF Vectorization  
```python
ngram_range = (1,3)
max_features = 100000
min_df = 3
sublinear_tf = True
````

**Why?**

* Suicide intent often appears in multi-word patterns
* Trigrams capture context better
* Larger vocabulary improves recall
* Sublinear TF improves stability

---

##  B. Hyperparameter Tuning (GridSearchCV)

I tuned:

* `C` (regularization strength)
* `penalty`
* `solver`

With:

```python
scoring = "f1_macro"
cv = 3
```

This ensures the model is optimized beyond default sklearn settings.

---

##  C. Class Imbalance Handling

* `class_weight="balanced"` enabled in Logistic Regression
* SMOTE support added (optional)

This prevents bias toward the majority class.

---

#  5. Project Structure

```
 Suicide-Ideation-Baseline
│
├── main.py         # main training pipeline
├── model.py        # model utilities
├── explainability.py 
├── viz_baseline.py 
├── Suicide_Detection.csv             # dataset
├── improved_baseline_lr.pkl          # saved model
├── improved_baseline_tfidf.pkl       # saved vectorizer
│
└── README.md
```

---

#  6. Installation

```bash
pip install -r requirements.txt
```

Main libraries:

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
joblib
```

---

#  7. Running the Model

### **Train + Evaluate**

```bash
python baseline_improved_main.py
```

### **Load Saved Model**

```python
import joblib

model = joblib.load("improved_baseline_lr.pkl")
tfidf  = joblib.load("improved_baseline_tfidf.pkl")
```

---

#  8. Final Evaluation Results

###  Hyperparameter Tuning

Best parameters found:

```python
{'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}
```

Best cross-validated F1 Score:

```
0.9393
```

---

##  Validation Set Performance

| Metric       | non-suicide | suicide  | Macro Avg |
| ------------ | ----------- | -------- | --------- |
| Precision    | 0.94        | 0.95     | 0.94      |
| Recall       | 0.95        | 0.94     | 0.94      |
| F1-score     | **0.94**    | **0.94** | **0.94**  |
| **Accuracy** | **0.9405**  |          |           |

**Support:** 37,132 samples

---

##  Test Set Performance

| Metric       | non-suicide | suicide  | Macro Avg |
| ------------ | ----------- | -------- | --------- |
| Precision    | 0.94        | 0.94     | 0.94      |
| Recall       | 0.94        | 0.94     | 0.94      |
| F1-score     | **0.94**    | **0.94** | **0.94**  |
| **Accuracy** | **0.9404**  |          |           |

**Support:** 46,415 samples

---

# 9. Summary of Contributions

My improved baseline model achieves:

* **94% accuracy**
* **0.94 macro F1**
* Perfectly balanced performance across classes
* Strong evidence that classical ML can still perform exceptionally well with optimized TF-IDF

This model is now a **solid anchor** for comparing advanced models like:

* BERT
* RoBERTa
* MentalBERT
* RAG-based approaches

---




