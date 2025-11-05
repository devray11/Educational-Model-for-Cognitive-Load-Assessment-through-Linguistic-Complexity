# Cognitive Load Assessment via Linguistic Analysis

## A Machine Learning Project for Text Difficulty Classification

This repository contains a comprehensive machine learning system designed to assess the cognitive load of educational texts. By analyzing a wide array of linguistic features, the system classifies content into specific difficulty levels (e.g., School, University, PhD), providing a valuable tool for personalizing learning experiences.

This project is based on the analysis of a large-scale dataset (`book_dataset.csv`) and involves the training and comparison of multiple machine learning models. The Random Forest classifier was identified as the optimal model, achieving high accuracy (98.75% F1-score) in predicting text difficulty.

The system is deployed as a simple web application consisting of a Flask backend API that serves the trained model, and an HTML/JavaScript frontend for user interaction.

## Features

* **Text Difficulty Classification:** Automatically classifies input text into multiple educational levels based on linguistic complexity.
* **Advanced Linguistic Analysis:** Extracts a comprehensive set of features, including established readability metrics (Flesch-Kincaid, SMOG), lexical sophistication, word concreteness, and sentence complexity.
* **High-Accuracy Ensemble Model:** Employs a trained Random Forest model (`cognitive_load_model.pkl`) for robust and accurate predictions.
* **REST API:** A lightweight backend API built with Flask (`App.py`) to handle text analysis requests.
* **Web Interface:** A simple, clean user interface (`index.html`) to paste text and receive an immediate classification and confidence score.

## Technologies Used

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **Machine Learning:** Scikit-learn (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)
* **NLP & Feature Extraction:** NLTK, textstat
* **Data Processing & Analysis:** Pandas, NumPy, Matplotlib, Seaborn
