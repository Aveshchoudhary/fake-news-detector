"""Train a fake-news classifier on bundled English + Hindi data."""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA = os.path.join(os.path.dirname(__file__),"sample_news.csv")
OUT = os.path.join(os.path.dirname(__file__),"model.joblib")


def build_pipeline():
    # Combine word + character n-grams so Devanagari (Hindi) is handled
    # alongside Latin (English) without needing a separate tokenizer.
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=0.95, sublinear_tf=True
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    # LogisticRegression gives calibrated predict_proba for confidence scores.
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def main():
    df = pd.read_csv(DATA)
    X, y = df["text"].astype(str), df["label"].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)
    print(classification_report(yte, pipe.predict(Xte)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump(pipe, OUT)
    print(f"Saved model -> {OUT}")


if __name__ == "__main__":
    main()
