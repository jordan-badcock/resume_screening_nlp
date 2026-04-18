import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def load_data():
    # Load the dataset
    df = pd.read_csv("./data/Resume/Resume.csv")

    # Drop the HTML column if it is there
    if "Resume_html" in df.columns:
        df = df.drop(columns=["Resume_html"])

    # Remove empty and duplicate resumes
    df = df.dropna(subset=["Resume_str"])
    df = df[df["Resume_str"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["Resume_str"])

    # Rename the columns so they are easier to work with
    df = df.rename(columns={"Resume_str": "text", "Category": "label"})

    print("Dataset summary")
    print(f"Rows: {len(df)}")
    print(f"Categories: {df['label'].nunique()}")
    print(df["label"].value_counts().sort_index().to_string())

    return df


def clean_texts(texts):
    # Clean the resume text with spaCy
    cleaned = []

    for doc in nlp.pipe(texts, batch_size=500):
        tokens = []
        for token in doc:
            # Remove spaces and punctuation
            if token.is_space or token.is_punct:
                continue
            # Remove stopwords
            if token.is_stop:
                continue
            # Remove URLs and emails
            if token.like_url or token.like_email:
                continue
            # Keep alphabetic words only
            if not token.is_alpha:
                continue

            # Lemmatize and lowercase
            tokens.append(token.lemma_.lower())

        cleaned.append(" ".join(tokens))

    return cleaned


def prepare_data(df):
    # Clean the resume text
    df["cleaned"] = clean_texts(df["text"].astype(str))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def run_model(model, X_train, X_test, y_train, y_test, name):
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the results
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    print(f"\n===== {name} =====")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    return {
        "name": name,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "report_text": report_text,
        "report_dict": report_dict,
    }


def print_summary(results):
    # Build a summary table for all experiments
    summary_df = pd.DataFrame(results)
    summary_df = summary_df[
        ["name", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ]
    summary_df = summary_df.sort_values(
        by=["accuracy", "macro_f1"], ascending=False
    ).reset_index(drop=True)

    print("\n===== Overall Summary =====")
    print(
        summary_df.to_string(
            index=False,
            header=[
                "Model",
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
            ],
            float_format=lambda value: f"{value:.4f}",
        )
    )


def print_best_model_details(best_result):
    # Convert the classification report to a dataframe so it can be sorted
    report_df = pd.DataFrame(best_result["report_dict"]).transpose()

    class_rows = report_df.loc[
        ~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    class_rows = class_rows.sort_values(by="f1-score")

    print(f"\nBest model: {best_result['name']}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    print(f"Best macro F1: {best_result['macro_f1']:.4f}")

    print(f"\n===== Detailed Results: {best_result['name']} =====")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(best_result["confusion_matrix"])

    print("\nLowest per-class F1 scores:")
    print(
        class_rows[["precision", "recall", "f1-score", "support"]]
        .head(5)
        .to_string(float_format=lambda value: f"{value:.4f}")
    )

    print("\nHighest per-class F1 scores:")
    print(
        class_rows[["precision", "recall", "f1-score", "support"]]
        .tail(5)
        .to_string(float_format=lambda value: f"{value:.4f}")
    )

    print("\nFull classification report:")
    print(best_result["report_text"])


def main():
    # Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    results = []

    # BoW unigram features
    bow_unigram = CountVectorizer(max_features=5000, ngram_range=(1, 1))
    X_train_bow_unigram = bow_unigram.fit_transform(X_train)
    X_test_bow_unigram = bow_unigram.transform(X_test)

    print(f"\nBoW unigram vocab size: {len(bow_unigram.vocabulary_)}")
    results.append(
        run_model(
            MultinomialNB(),
            X_train_bow_unigram,
            X_test_bow_unigram,
            y_train,
            y_test,
            "Naive Bayes (BoW Unigrams)",
        )
    )
    results.append(
        run_model(
            LinearSVC(max_iter=2000, random_state=42),
            X_train_bow_unigram,
            X_test_bow_unigram,
            y_train,
            y_test,
            "SVM (BoW Unigrams)",
        )
    )
    results.append(
        run_model(
            LogisticRegression(max_iter=2000, random_state=42),
            X_train_bow_unigram,
            X_test_bow_unigram,
            y_train,
            y_test,
            "Logistic Regression (BoW Unigrams)",
        )
    )

    # BoW unigram + bigram features
    bow_bigram = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_bow_bigram = bow_bigram.fit_transform(X_train)
    X_test_bow_bigram = bow_bigram.transform(X_test)

    print(f"\nBoW unigram+bigram vocab size: {len(bow_bigram.vocabulary_)}")
    results.append(
        run_model(
            MultinomialNB(),
            X_train_bow_bigram,
            X_test_bow_bigram,
            y_train,
            y_test,
            "Naive Bayes (BoW Unigrams+Bigrams)",
        )
    )
    results.append(
        run_model(
            LinearSVC(max_iter=2000, random_state=42),
            X_train_bow_bigram,
            X_test_bow_bigram,
            y_train,
            y_test,
            "SVM (BoW Unigrams+Bigrams)",
        )
    )
    results.append(
        run_model(
            LogisticRegression(max_iter=2000, random_state=42),
            X_train_bow_bigram,
            X_test_bow_bigram,
            y_train,
            y_test,
            "Logistic Regression (BoW Unigrams+Bigrams)",
        )
    )

    # TF-IDF unigram features
    tfidf_unigram = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    X_train_tfidf_unigram = tfidf_unigram.fit_transform(X_train)
    X_test_tfidf_unigram = tfidf_unigram.transform(X_test)

    print(f"\nTF-IDF unigram vocab size: {len(tfidf_unigram.vocabulary_)}")
    results.append(
        run_model(
            MultinomialNB(),
            X_train_tfidf_unigram,
            X_test_tfidf_unigram,
            y_train,
            y_test,
            "Naive Bayes (TF-IDF Unigrams)",
        )
    )
    results.append(
        run_model(
            LinearSVC(max_iter=2000, random_state=42),
            X_train_tfidf_unigram,
            X_test_tfidf_unigram,
            y_train,
            y_test,
            "SVM (TF-IDF Unigrams)",
        )
    )
    results.append(
        run_model(
            LogisticRegression(max_iter=2000, random_state=42),
            X_train_tfidf_unigram,
            X_test_tfidf_unigram,
            y_train,
            y_test,
            "Logistic Regression (TF-IDF Unigrams)",
        )
    )

    # TF-IDF unigram + bigram features
    tfidf_bigram = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf_bigram = tfidf_bigram.fit_transform(X_train)
    X_test_tfidf_bigram = tfidf_bigram.transform(X_test)

    print(f"\nTF-IDF unigram+bigram vocab size: {len(tfidf_bigram.vocabulary_)}")
    results.append(
        run_model(
            MultinomialNB(),
            X_train_tfidf_bigram,
            X_test_tfidf_bigram,
            y_train,
            y_test,
            "Naive Bayes (TF-IDF Unigrams+Bigrams)",
        )
    )
    results.append(
        run_model(
            LinearSVC(max_iter=2000, random_state=42),
            X_train_tfidf_bigram,
            X_test_tfidf_bigram,
            y_train,
            y_test,
            "SVM (TF-IDF Unigrams+Bigrams)",
        )
    )
    results.append(
        run_model(
            LogisticRegression(max_iter=2000, random_state=42),
            X_train_tfidf_bigram,
            X_test_tfidf_bigram,
            y_train,
            y_test,
            "Logistic Regression (TF-IDF Unigrams+Bigrams)",
        )
    )

    # Sort the experiments from best to worst
    results = sorted(
        results,
        key=lambda result: (result["accuracy"], result["macro_f1"]),
        reverse=True,
    )

    print_summary(results)
    print_best_model_details(results[0])


if __name__ == "__main__":
    main()
