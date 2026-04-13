#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

df=pd.read_csv("D:\meet\Machine-learning-programs-main\Machine-learning-programs-main\Datasets\titanic.csv")
df.head()

#dropping irrelevant columns
df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

df.columns

count=df.isnull().sum()
count

## Handling missing values
age_imputer = SimpleImputer(strategy='median')
emb_imputer = SimpleImputer(strategy='most_frequent')

df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()
df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()

#Encoding Categorical Variables
label_encoder=LabelEncoder()
df['Sex']=label_encoder.fit_transform(df['Sex'])
df['Embarked']=label_encoder.fit_transform(df['Embarked'])

#spliting data into features and target
X=df.drop('Survived',axis=1)
Y=df['Survived']

#spliting into train test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#standardizing features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#intializaing logistic regression model
log_reg_model=LogisticRegression()

# Training the model
log_reg_model.fit(X_train, Y_train)

#making predictions
Y_pred=log_reg_model.predict(X_test)

#Evaluating the model
print("Accuracy=",accuracy_score(Y_test,Y_pred))

print("\nClassification Report :")
print(classification_report(Y_test,Y_pred))

#svm model creating
svm_model=SVC(kernel='linear',random_state=42)

#train model
svm_model.fit(X_train,Y_train)

#Prediction
svm_pred=svm_model.predict(X_test)

#Evaluate SVM
print("\nSVM Accuracy:",accuracy_score(Y_test,svm_pred))
print("\nSVM Classification Report :\n",classification_report(Y_test,svm_pred))

#compare both model
print("\n---Model Comparision---")
print("Logistic Regrassion Accuracy :",accuracy_score(Y_test,Y_pred))
print("SVM Accuracy :",accuracy_score(Y_test,svm_pred))
