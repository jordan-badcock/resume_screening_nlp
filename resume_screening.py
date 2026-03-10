import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Same pattern as your labs: load spaCy model with parser/ner disabled for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_texts(texts):
    """
    Scope-safe cleaning: mirrors your Lab 3 approach (remove punctuation/stopwords, lowercase)
    and can be extended like your Week 4 lab (urls/emails, lemmatization) if you want.
    """
    docs = nlp.pipe(texts, batch_size=500)
    cleaned = []

    for doc in docs:
        tokens = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            if t.is_stop:
                continue
            if t.like_url or t.like_email:
                continue
            if not t.is_alpha:
                continue
            tokens.append(t.lemma_.lower())
        cleaned.append(" ".join(tokens))

    return cleaned

def load_data():
    df = pd.read_csv("./data/Resume/Resume.csv")
    print(f"{df.head()}\n{df.shape}")

    # Drop Resume_html column
    df = df.drop(columns=["Resume_html"])
    # Remove empties and duplicates
    df = df.dropna(subset=["Resume_str"])
    df = df.drop_duplicates(subset=["Resume_str"])

    df = df.rename(columns={"Resume_str": "text", "Category": "label"})

    return df

def vectorize_and_split(df):
    df["cleaned"] = clean_texts(df["text"].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"]
    )

    bow_vectorizer = CountVectorizer(max_features=5000)
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_bow, X_test_bow, X_train_tfidf, X_test_tfidf, y_train, y_test

def run_model(model, X_train, X_test, y_train, y_test, name="MODEL"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix (rows=true, cols=pred):\n{cm}")

    # This is okay to include (you used similar reporting in your Week 4 lab)
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

def main():
    df = load_data()
    X_train_bow, X_test_bow, X_train_tfidf, X_test_tfidf, y_train, y_test = vectorize_and_split(df)

    # BoW models
    run_model(MultinomialNB(), X_train_bow, X_test_bow, y_train, y_test, "Naive Bayes (BoW)")
    run_model(LinearSVC(max_iter=2000, random_state=42), X_train_bow, X_test_bow, y_train, y_test, "SVM (BoW)")
    run_model(LogisticRegression(max_iter=2000, random_state=42), X_train_bow, X_test_bow, y_train, y_test, "Logistic Regression (BoW)")

    # TF-IDF models
    run_model(MultinomialNB(), X_train_tfidf, X_test_tfidf, y_train, y_test, "Naive Bayes (TF-IDF)")
    run_model(LinearSVC(max_iter=2000, random_state=42), X_train_tfidf, X_test_tfidf, y_train, y_test, "SVM (TF-IDF)")
    run_model(LogisticRegression(max_iter=2000, random_state=42), X_train_tfidf, X_test_tfidf, y_train, y_test, "Logistic Regression (TF-IDF)")

if __name__ == "__main__":
    main()