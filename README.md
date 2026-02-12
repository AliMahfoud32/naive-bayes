# naive-bayes
Naive Bayes classification project implemented in Python. This model applies Bayes’ Theorem with the assumption of feature independence to perform efficient probabilistic classification. Includes data preprocessing, model training with scikit-learn, prediction, and performance evaluation for practical ML tasks.
Project Overview

This project demonstrates the implementation of the Naive Bayes algorithm for supervised classification using Python. The goal is to understand both the mathematical foundation of probabilistic models and their practical implementation using real data.

Naive Bayes is a fast and efficient algorithm widely used in text classification, spam detection, sentiment analysis, and other high-dimensional problems.
The model assumes that features are conditionally independent given the class label.
Although this assumption is “naive,” the algorithm performs surprisingly well in many real-world scenarios.

To improve numerical stability, log probabilities are used during computation.

Technologies Used
Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib / Seaborn
Results

The model was evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

The results demonstrate that Naive Bayes is effective and computationally efficient for this classification task.

What I learnt:

Understanding probabilistic machine learning models

Applying Bayes’ Theorem in practice

Working with scikit-learn classifiers

Evaluating model performance properly

Handling numerical stability using log probabilities

Future Improvements:

Implement Naive Bayes from scratch (without scikit-learn)

Compare performance with Logistic Regression

Add cross-validation

Perform hyperparameter tuning

Test on larger or different datasets

