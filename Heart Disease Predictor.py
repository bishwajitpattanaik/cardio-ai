#continued from Heart.py

df_encoded


df_encoded.columns
# Index(['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
#        'HeartDisease', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP',
#        'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST',
#        'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'],
#       dtype='object')



#import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split  

#import StandardScaler for feature scaling (normalizing data)
from sklearn.preprocessing import StandardScaler  

#import evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report  

#import machine learning models
from sklearn.linear_model import LogisticRegression   # Logistic Regression model
from sklearn.naive_bayes import GaussianNB            # Naive Bayes model
from sklearn.tree import DecisionTreeClassifier       # Decision Tree model
from sklearn.svm import SVC                           # Support Vector Machine model
from sklearn.neighbors import KNeighborsClassifier    # KNN model




#selecting input and output columns

# X contains all columns except target column 'HeartDisease'
X = df_encoded.drop('HeartDisease', axis=1)

# y contains only target/output column
# 0 = No heart disease, 1 = Heart disease
y = df_encoded['HeartDisease']



#Splitting data

#split dataset into training and testing data
# test_size=0.2 means 20% data for testing, 80% for training
# stratify=y keeps same class ratio in train and test
# random_state=42 ensures same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


#feature scaling

#create scaler object
scaler = StandardScaler()

#fit scaler on training data and transform it
#makes mean=0 and std=1
X_train_scaled = scaler.fit_transform(X_train)

#apply same scaling to test data
X_test_scaled = scaler.transform(X_test)




#creating multiple models

#store multiple ML models inside dictionary
models = {
    'Logistic Regression': LogisticRegression(),      # Linear classification
    'K NeighborsClassifier': KNeighborsClassifier(),  # Distance-based model
    'Naive Bayes': GaussianNB(),                      # Probability-based model
    'Decision Tree': DecisionTreeClassifier(),        # Tree-based model
    'SVM (RBF kernel)': SVC(probability=True)         # SVM with probability output
}
#dictionary format:
#key = model name
#value = model object
#this helps run all models in one loop.



# Empty list to save model results
result = []




#training and evaluating all models

#loop through each model in dictionary
for name, model in models.items():

    # Train model using training data
    model.fit(X_train_scaled, y_train)

    # Predict output for test data
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    # Store model results in result list as dictionary
    result.append({
        'Model': name,
        'Accuracy': round(acc, 4),   # round to 4 decimal places
        'F1 Score': round(f1, 4)
    })

#flow inside loop:
# Train model
# Predict
# Accuracy calculate
# F1 score calculate
# Save results


#print all results
# result
# [{'Model': 'Logistic Regression', 'Accuracy': 0.875, 'F1 Score': 0.8878},
#  {'Model': 'KNN', 'Accuracy': 0.8859, 'F1 Score': 0.8986},
#  {'Model': 'Naive Bayes', 'Accuracy': 0.8696, 'F1 Score': 0.8788},
#  {'Model': 'Decision Tree', 'Accuracy': 0.75, 'F1 Score': 0.7604},
#  {'Model': 'SVM (RBF Kernel)', 'Accuracy': 0.8641, 'F1 Score': 0.8804}]


#pickle module in Python is used for serializing and deserializing Python objects.

#Serialization, also known as pickling, converts a Python object into a byte stream.
#this byte stream can then be stored in a file, sent over a network, or saved in a database.

# Deserialization, also known as unpickling, is the process of converting
# the byte stream back into a Python object.

#here we want to use knn model as it has the best accuracy among all models
#so we have to save the knn model 
#we can do this by using pickle module as we can assume model as an object
#we have to save knn model, knn scalar object, and list of columns

#pip install joblib
#import joblib library
#used to save and load trained machine learning models efficiently
import joblib

#save the trained KNN model into a file named 'KNN_heart.pkl'
#models['KNN'] accesses the KNN model from the models dictionary
joblib.dump(models['KNN'], 'KNN_heart.pkl')

#save the fitted scaler object into a file
#this scaler will be reused later to scale new input data
joblib.dump(scaler, 'scaler.pkl')

#save list of feature/column names into a file
#X.columns.tolist() converts dataframe column names into a Python list
#needed later to match input columns during prediction
joblib.dump(X.columns.tolist(), 'columns.pkl')

#but using no .pkl files for this project 
#instead, the model trains on startup from heart.csv using @st.cache_resource
#version-safe across all Python/sklearn environments






