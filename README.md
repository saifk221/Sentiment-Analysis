# ML Projects

Two scikit-learn projects covering the core supervised learning workflow: regression on tabular data and classification on text.

## Projects

### 1. California Housing — Linear Regression

Predicts median house value from California census features. Compares simple linear regression (single feature) against multiple linear regression (all features) on the 20,640-sample sklearn California housing dataset.

**File:** [`california_housing_regression.py`](california_housing_regression.py)

**Results (test set, 20% holdout):**

| Model                       | R²   | MSE         |
| --------------------------- | ---- | ----------- |
| Simple (MedInc only)        | 0.48 | 7.14 × 10⁹  |
| Multiple (all 8 features)   | 0.60 | 5.44 × 10⁹  |

Adding the remaining features lifts R² by ~0.12 over median income alone, confirming that location and housing-stock attributes carry predictive signal independent of income.

### 2. 20 Newsgroups — Text Classification

Classifies newsgroup posts into one of four topics (`rec.autos`, `rec.sport.baseball`, `sci.electronics`, `talk.politics.misc`) using TF-IDF features with a Linear SVM and Multinomial Naive Bayes.

**File:** [`text_classification.py`](text_classification.py)

**Pipeline:**
1. Light text cleaning — strip emails, headers (`From:`, `Subject:`, etc.), and non-word characters
2. TF-IDF vectorization with English stop words, capped at 10k features
3. Train `LinearSVC` and `MultinomialNB`
4. Evaluate on the held-out test split

**Results:**

| Model           | Accuracy | F1 (weighted) |
| --------------- | -------- | ------------- |
| LinearSVC       | 0.937    | 0.937         |
| MultinomialNB   | 0.939    | 0.939         |

Both models perform comparably on this 4-class problem. The script also runs both classifiers on a few hand-written sample sentences for a qualitative sanity check.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python california_housing_regression.py
python text_classification.py
```

Both scripts download their datasets automatically through scikit-learn on first run.

## Stack

Python · scikit-learn · NumPy · pandas · matplotlib
