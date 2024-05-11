# Author: Aakarshit Srivastava
# Team: Perfect Cube
# Contact: Shubh26002@gmail.com

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def evaluation(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Model Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    evaluation(y_test, y_pred)

# Step 1: Dataset Analysis and Preprocessing

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# New Data
# data = pd.read_csv('IBMEmployee_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop irrelevant columns (if any)
data = data.drop(columns=['EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'])

# Encode categorical variables using One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop(columns=['Attrition_Yes'])
y = data['Attrition_Yes']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Development

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


# Initialize and train Logistic Regression classifier
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train, y_train)

# Initialize and train Support Vector Machine classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)

# Initialize and train XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Initialize and train AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(random_state=42)
adaboost_classifier.fit(X_train, y_train)

# Initialize and train Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Initialize and Train Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Initialize and train KNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

#  Initialize and train Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)

# Initialize and train Neural Network classifier
nn_classifier = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

nn_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn_classifier.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)


# Step 3: Model Evaluation and Optimization

# Evaluate Random Forest Classifier model
print("\nRandom Forest Classifier Model Performance:")
evaluate_model(rf_classifier, X_test, y_test)

# Evaluate Logistic Regression model
print("Logistic Regression Model Performance:")
evaluate_model(logistic_classifier, X_test, y_test)

# Evaluate Support Vector Machine model
print("\nSupport Vector Machine Model Performance:")
evaluate_model(svm_classifier, X_test, y_test)

# Evaluate XGBoost model
print("\nXGBoost Model Performance:")
evaluate_model(xgb_classifier, X_test, y_test)

# Evaluate AdaBoost model
print("\nAdaBoost Model Performance:")
evaluate_model(adaboost_classifier, X_test, y_test)

# Evaluate Decision Tree model
print("\nDecision Tree Model Performance:")
evaluate_model(dt_classifier, X_test, y_test)

# Evaluate Naive Bayes Classifier
print("Naive Bayes Model Performance:")
nb_eval=evaluate_model(nb_classifier, X_test, y_test)

# Evaluate KNN model
print("\nK-Nearest Neighbors Model Performance:")
evaluate_model(knn_classifier, X_test, y_test)

# Evaluate Gradient Boosting model
print("\nGradient Boosting Model Performance:")
evaluate_model(gb_classifier, X_test, y_test)

# Evaluate Neural Network model
print("\nNeural Network Model Performance:")
loss, accuracy = nn_classifier.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

