# Text classification using SVM and Naive Bayes
# Uses sklearn's built-in 20 Newsgroups dataset (~18,000 documents)

import re
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report


# Load data (4 categories to keep it focused)
categories = [
    "rec.sport.baseball",
    "sci.electronics",
    "talk.politics.misc",
    "rec.autos",
]

train_data = fetch_20newsgroups(subset="train", categories=categories, random_state=10)
test_data = fetch_20newsgroups(subset="test", categories=categories, random_state=10)

print(f"Training samples: {len(train_data.data)}")
print(f"Test samples:     {len(test_data.data)}")
print(f"Categories:       {train_data.target_names}")
print()

# Clean text (remove special chars, emails, headers)
def clean_text(text):
    text = re.sub(r"\S+@\S+", "", text)          # remove emails
    text = re.sub(r"^(From|Subject|Organization|Lines):.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\W+", " ", text)              # remove special chars
    return text.strip()

train_texts = [clean_text(doc) for doc in train_data.data]
test_texts = [clean_text(doc) for doc in test_data.data]

print("Sample document (cleaned):")
print(train_texts[0][:200], "...")
print(f"Label: {train_data.target_names[train_data.target[0]]}")
print()

# Tokenize
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
vec_train_x = vectorizer.fit_transform(train_texts)
vec_test_x = vectorizer.transform(test_texts)

# SVC model
svc_model = LinearSVC()
svc_model.fit(vec_train_x, train_data.target)

svc_pred = svc_model.predict(vec_test_x)
svc_accuracy = np.mean(svc_pred == test_data.target)
svc_f1 = f1_score(test_data.target, svc_pred, average="weighted")

print("--- SVC Results ---")
print(f"Accuracy: {svc_accuracy:.4f}")
print(f"F1 Score: {svc_f1:.4f}")
print()

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(vec_train_x, train_data.target)

nb_pred = nb_model.predict(vec_test_x)
nb_accuracy = np.mean(nb_pred == test_data.target)
nb_f1 = f1_score(test_data.target, nb_pred, average="weighted")

print("--- Naive Bayes Results ---")
print(f"Accuracy: {nb_accuracy:.4f}")
print(f"F1 Score: {nb_f1:.4f}")
print()

# Detailed classification report (best model)
print("--- Classification Report (SVC) ---")
print(classification_report(test_data.target, svc_pred, target_names=train_data.target_names))

# Try it on custom text
samples = [
    "The pitcher threw a no-hitter in last night's game and the crowd went wild",
    "The new GPU from NVIDIA delivers twice the performance of the previous generation",
    "The senator proposed a new bill to reform campaign finance regulations",
    "The new sports car has a turbocharged V8 engine with 600 horsepower",
]

print("--- Custom Predictions ---")
for text in samples:
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    svc_label = train_data.target_names[svc_model.predict(vec)[0]]
    nb_label = train_data.target_names[nb_model.predict(vec)[0]]
    print(f'Text: "{text}"')
    print(f"  SVC: {svc_label}")
    print(f"  NB:  {nb_label}")
    print()
