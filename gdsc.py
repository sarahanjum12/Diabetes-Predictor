import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle
from flask import Flask, jsonify, request, render_template
import requests
import json
import numpy as np


# Load the dataset into a pandas dataframe
df = pd.read_csv("C:\Dataset of Diabetes .csv")
print(df.head())

# Preprocess the data

# Drop irrelevant columns such as ID and No. of Patient
df = df.drop(['ID', 'No_Pation'], axis=1)

# Handle missing values

print(df.columns)


# Convert categorical variables such as Gender and Class to numerical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'CLASS'])
print(df.head())
# Split the dataset into training and testing sets
X = df.drop('CLASS_N', axis=1)  # Features
y = df['CLASS_N']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a machine learning algorithm
model = LogisticRegression()

# Train the machine learning algorithm
model.fit(X_train, y_train)
feature_names = X.columns.tolist()
print(feature_names)

predictions = model.predict(X_test)
print(predictions)
#Measure Performance of the model
from sklearn.metrics import classification_report

#Measure performance of the model
classification_report(y_test, predictions)
print(classification_report)
print(df.head())


#print the evaluation of training data 
print("The evaluation of training data\n")
y_train_pred = model.predict(X_train)
print('Training Accuracy:', accuracy_score(y_train, y_train_pred))
print('Training Precision:', precision_score(y_train, y_train_pred))
print('Training Recall:', recall_score(y_train, y_train_pred))
print('Training F1-score:', f1_score(y_train, y_train_pred))

# Evaluate the performance of the trained model on the testing data
y_pred = model.predict(X_test)
print("The evaluation on testing data/n")
print('Testing Accuracy:', accuracy_score(y_test, y_pred))
print('Testing Precision:', precision_score(y_test, y_pred))
print('Testing Recall:', recall_score(y_test, y_pred))
print('Testing F1-score:', f1_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Plot the confusion matrix
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


joblib.dump(model, 'diabetes_model.joblib')

app=Flask(__name__)

@app.route('/info',methods=['GET','POST'])
def data():
    if request.method == 'POST':
        ID = request.form['ID']
        No_Pation = request.form['No_Pation']
        Gender=request.form['Gender']
        AGE = request.form['AGE']
        Urea = request.form['Urea']
        HbA1c=request.form['HbA1c']
        Chol = request.form['Chol']
        TG = request.form['TG']
        HDL=request.form['HDL']
        LDL = request.form['LDL']
        VLDL= request.form['VLDL']
        BMI=request.form['BMI']
        CLASS_N=request.form['CLASS_N']
        CLASS_N1=request.form['CLASS_N']
        CLASS_P= request.form['CLASS_P']
        CLASS_Y=request.form['CLASS_Y']
        data = dict(request.form)
        model = joblib.load('diabetes_model.joblib')
        data_list =list(data.values())
        a = np.array(data_list,dtype=float)
        c=a.reshape(1,-1)
        
        
       
        predictions = model.predict(c)
        return f'''
            <html>
                <head>
                    <title>Input Data and Diabetes Prediction</title>
                </head>
                <body>
                    <h2>Input Data and Diabetes Prediction:</h2>
                    <p>ID: {ID}</p>
                    <p>No_Pation: {No_Pation}</p>
                    <p>Gender: {Gender}</p>
                    <p>AGE: {AGE}</p>
                    <p>Urea: {Urea}</p>
                    <p>HbA1c: {HbA1c}</p>
                    <p>Chol: {Chol}</p>
                    <p>TG: {TG}</p>
                    <p>HDL: {HDL}</p>
                    <p>LDL: {LDL}</p>
                    <p>VLDL: {VLDL}</p>
                    <p>BMI: {BMI}</p>
                    <p>PREDICTION: If prediction is {predictions} or 1 You are diabetic</p>
                    <p>PREDICTION: Else if prediction is {predictions} or 0 You are Not Diabetic</p>
                </body>
            </html>
        '''
    return '''
        <html>
            <head>
                <title>Input Data and Diabetes Prediction</title>
            </head>
            <body>
                <form method="post">
                    <label for="ID">ID:</label>
                    <input type="number" id="ID" name="ID"><br>

                    <label for="No_Pation">No_Pation:</label>
                    <input type="number" id="No_Pation" name="No_Pation"><br>

                    <label for="Gender">Gender enter 1 if male and 0 if female:</label>
                    <input type="number" id="Gender" name="Gender"><br>

                    <label for="AGE">AGE:</label>
                    <input type="number" id="AGE" name="AGE"><br>

                    <label for="Urea">Urea:</label>
                    <input type="number" id="Urea" name="Urea"><br>

                    <label for="Cr">Cr:</label>
                    <input type="number" id="Cr" name="Cr"><br>

                    <label for="HbA1c">HbA1c:</label>
                    <input type="number" id="HbA1c" name="HbA1c"><br>

                    <label for="Chol">Chol:</label>
                    <input type="number" id="Chol" name="Chol"><br>

                    <label for="TG">TG:</label>
                    <input type="number" id="TG" name="TG"><br>

                    <label for="HDL">HDL:</label>
                    <input type="number" id="HDL" name="HDL"><br>

                    <label for="LDL">LDL:</label>
                    <input type="number" id="LDL" name="LDL"><br>

                    <label for="VLDL">VLDL:</label>
                    <input type="number" id="VLDL" name="VLDL"><br>

                    <label for="BMI">BMI:</label>
                    <input type="number" id="BMI" name="BMI"><br>

                    <label for="CLASS_N">CLASS_N:</label>
                    <input type="number" id="CLASS_N" name="CLASS_N"><br>

                    <label for="CLASS_N1">CLASS_N1 enter 0 or 1:</label>
                    <input type="number" id="CLASS_N1" name="CLASS_N1"><br>

                    <label for="CLASS_P">CLASS_P:</label>
                    <input type="number" id="CLASS_P" name="CLASS_P"><br>

                    <label for="CLASS_Y">CLASS_Y:</label>
                    <input type="number" id="CLASS_Y" name="CLASS_Y"><br>

                     <input type="submit" value="Submit">
                </form>
              </body>
          </html>
    '''
if __name__ == '__main__':
    app.run(debug=True)

        



