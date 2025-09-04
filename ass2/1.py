from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split

#load dataset
iris = load_iris()
X = iris.data
y = (iris.target==0).astype(int)

#split data into training and testing
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42,stratify=y
)

#to train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

#make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)

#print results
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)