

# GOAL --> DETECT DDOS TRAFFIC

#=============================================@ANOMIX_project.ipynb



#data handling and preprocessing
import numpy as np
import pandas as pd

#data scaling
from sklearn.preprocessing import StandardScaler 

#model training and evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#graph
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


df = pd.read_csv("D:\onedrive\Desktop\CIC DDOS 2017 DATA_SET\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
   
df.head()  
#df['Destination Port']
df.info()
   
X = df
X = X.drop(' Label',axis= 1)
   
y = df.iloc[:,78]  
y

   
X.info()  
X = X.fillna(X['Flow Bytes/s'].mean())
X.info()  
print(np.isinf(X).sum()) 
print(np.isnan(X).sum())   
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace =True)


bad_rows = X[np.isnan(X).any(axis = 1) | np.isinf(X).any(axis=1)] 
bad_rows 
y.info
y_ = type(y)
y_

   
#split data into training and testing

x_train, x_test, y_train, y_test =train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 4000)
y_test = y_test.reset_index(drop = True)


y_test.shape

   
# data encoding using label encoder

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(type(x_train),type(y_train))

scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = pd.DataFrame(x_train)

plt.scatter(x_train.iloc[:,0], y_train,alpha=0.3)
plt.xlabel('feature 1')
plt.ylabel('target')
plt.show()

   
print()

   
# shuffle the y target to test model validity

#y_train = np.random.permutation(y_train)


   
#model training and testing

model = LogisticRegression(max_iter = 500000)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_pred

   


   
#accuracy score and classification report

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test,y_pred)

print("accuracy score: ",accuracy)
print("classification report: \n",classification_report(y_test, y_pred))


   
#RocCurve 

RocCurveDisplay.from_estimator(model, x_test, y_test)
plt.show()

   
#confusion matrix

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

   
#find std deviations between the five different cross value scores

from sklearn.model_selection import cross_val_score 
scores = cross_val_score(model,x_test, y_test, cv =5)


print("mean score of 5 data splits: ",scores.mean())
print("scores: ",scores)
print("std: ",scores.std())

   
#bad_rows = np.sum(X[np.isnan(X).any(axis = 1) | np.isinf(X).any(axis=1)])
#bad_rows

   
y_test.astype(int)

   
type(x_test)

   



