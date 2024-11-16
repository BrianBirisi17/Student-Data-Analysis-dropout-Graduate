import pandas as pd
import joblib as jb
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

myData=pd.read_csv("/content/drive/MyDrive/studiesreports.csv")
mappingTargetData={"Dropout":0,"Graduate":1}
myData["Target"]=myData["Target"].map(mappingTargetData)
myData['Target'] = myData['Target'].replace(['NaN', 'None', ''], pd.NA)
myData['Target'] = pd.to_numeric(myData['Target'], errors='coerce')
myDataCleaned = myData.dropna(subset=['Target'])
myDataCleaned['Target'] = myDataCleaned['Target'].astype(int)
myData=myDataCleaned
X=myData.drop(columns=["Application mode","Nacionality","Target","Displaced"])
y=myData["Target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #Here we are training  the model
DTC=DecisionTreeClassifier(criterion="entropy", random_state=42)
DTC.fit(X_train,y_train)
y_pred=DTC.predict(X_test)
print(y_pred)

accuracy_score=DTC.score(X_test,y_test)
report=classification_report(y_test,y_pred)
cf=confusion_matrix(y_test,y_pred)
print(cf)
print(report)
print(accuracy_score)

jb.dump(DTC,"myModel.joblib")