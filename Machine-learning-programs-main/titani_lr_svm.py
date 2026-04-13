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


df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

df.columns

count=df.isnull().sum()
count


age_imputer = SimpleImputer(strategy='median')
emb_imputer = SimpleImputer(strategy='most_frequent')

df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()
df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()


label_encoder=LabelEncoder()
df['Sex']=label_encoder.fit_transform(df['Sex'])
df['Embarked']=label_encoder.fit_transform(df['Embarked'])


X=df.drop('Survived',axis=1)
Y=df['Survived']


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


log_reg_model=LogisticRegression()


log_reg_model.fit(X_train, Y_train)


Y_pred=log_reg_model.predict(X_test)


print("Accuracy=",accuracy_score(Y_test,Y_pred))

print("\nClassification Report :")
print(classification_report(Y_test,Y_pred))


svm_model=SVC(kernel='linear',random_state=42)


svm_model.fit(X_train,Y_train)


svm_pred=svm_model.predict(X_test)


print("\nSVM Accuracy:",accuracy_score(Y_test,svm_pred))
print("\nSVM Classification Report :\n",classification_report(Y_test,svm_pred))


print("\n---Model Comparision---")
print("Logistic Regrassion Accuracy :",accuracy_score(Y_test,Y_pred))
print("SVM Accuracy :",accuracy_score(Y_test,svm_pred))
