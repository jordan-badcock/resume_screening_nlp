# Resume Screening NLP

This project builds a resume-category classification pipeline using spaCy for text cleaning and scikit-learn for feature extraction and supervised learning. It compares multiple classifiers with both bag-of-words and TF-IDF representations.

## Repository Contents

- `resume_screening.py`: Main training and evaluation script.
- `data/Resume/Resume.csv`: Resume dataset used by the model.
- `data/data/data/`: Large collection of categorized resume PDF files.
- `NLP Project Proposal - Resume Screening.pdf`: Project proposal.
- `NLP Project - Progress Report.pdf`: Progress report.

## Pipeline Overview

The script currently:

1. Loads `data/Resume/Resume.csv`.
2. Drops the `Resume_html` column.
3. Removes empty and duplicate resume entries.
4. Renames fields to `text` and `label`.
5. Cleans text with spaCy by removing punctuation, stopwords, URLs, emails, non-alphabetic tokens, and by lemmatizing tokens.
6. Splits the cleaned data into train and test sets with stratification.
7. Builds both bag-of-words and TF-IDF features with a maximum vocabulary size of 5000.
8. Trains and evaluates:
   - Multinomial Naive Bayes
   - Linear SVM
   - Logistic Regression
9. Prints accuracy, confusion matrices, and classification reports.

## Requirements

- Python 3
- `pandas`
- `spacy`
- `scikit-learn`
- spaCy English model: `en_core_web_sm`

## Run

Install dependencies and the spaCy model, then run:

```bash
python resume_screening.py
```

## Data Notes

- `Resume.csv` contains the text and category labels used by the classifier.
- The repository also includes a large corpus of categorized resume PDFs, which may be useful for extension work or alternate preprocessing pipelines.

## Notes

- A local `.venv/` exists in the folder but is ignored by git.
- Because the repo includes substantial dataset content, pushing and cloning this project will be slower than a code-only repository.
