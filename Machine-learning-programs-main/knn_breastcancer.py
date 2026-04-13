from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
X=data.data
Y=data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

for k in range(1,8):
    

    classifier=KNeighborsClassifier(n_neighbors=k)

    
    classifier.fit(X_train,Y_train)

   
    y_pred=classifier.predict(X_test)

     
    cm=confusion_matrix(Y_test,y_pred)
    print(cm)
    accuracy=accuracy_score(Y_test,y_pred)*100

    print("Accuracy of our model is equal "+str(round(accuracy,2))+" %.")
    
