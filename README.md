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

Code:
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_table('SMSSpamCollection.txt',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Lets check the first 5 rows
print("Dataset shape: ", df.shape)
df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() # returns (rows, columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'],
                                                    test_size=0.2, 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
# The fit_transform method learns the vocabulary dictionary; i.e., it registers every vocabulary word found in the training data, then it will return document-term matrix.
training_data = count_vector.fit_transform(X_train)
print("Number of unique words in the training dataset:", training_data.shape[1])
# Alternatively:
# count_vector.fit(X_train)
# training_data = count_vector.transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
# Transform will return document-term matrix
# If a word is not found in the dictionary of the word dictionary (saved during fit method)
testing_data = count_vector.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print('Accuracy score: ', format(round(100*accuracy,2)),'%')
import seaborn as sns
# count plot on single categorical variable
ax = sns.countplot(x = df['label'], palette = 'rocket', hue = df['label'])

#add data labels
ax.bar_label(ax.containers[0])

# add plot title
plt.title("Observations by Classification Type")

# show plot
plt.show()
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, ConfusionMatrixDisplay

print('Precision score: ', format(round(100*precision_score(y_test, predictions),2)),'%')
print('Recall score: ',format(round(100*recall_score(y_test, predictions),2)),'%')
print('F1 score: ', format(round(100*f1_score(y_test, predictions),2)),'%')

#evaluate model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
    

#compute the confusion matrix.
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=naive_bayes.classes_)
disp.plot()
plt.show()
