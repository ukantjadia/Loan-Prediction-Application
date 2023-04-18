import pandas as pd 
# import pickle
import joblib
## Preparing model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
## Preparing encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
## Collecting the dataset
URL = "https://raw.githubusercontent.com/ukantjadia/Loan-Prediction/Main/Data/train_u6lujuX_CVtuZ9i.csv"
df = pd.read_csv(URL)
# df = pd.read_csv("../Data/train_u6lujuX_CVtuZ9i.csv")
## Droping the Loan id 
df.drop('Loan_ID',axis=1,inplace=True)

## Converting the data type of credit_history int to object
df['Credit_History'] = df['Credit_History'].astype('O')

# %%
## Seprating the data types
Tcat_data = []
Tnum_data = []
for name, dtype in enumerate(df.dtypes):
    if dtype == object:
        Tcat_data.append(df.iloc[:,name])
    else:
        Tnum_data.append(df.iloc[:,name])
Tcat_data = pd.DataFrame(Tcat_data).T
Tnum_data = pd.DataFrame(Tnum_data).T

## Handling the missing value in categorical and numerical data
### categorical data
Tcat_data = Tcat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
### Numerical data
Tnum_data.fillna(method='bfill',inplace=True)

## Transforming the categorical data
target_values = {'Y':0,'N':1}
target = Tcat_data['Loan_Status']
Tcat_data.drop('Loan_Status',axis=1,inplace=True)
target = target.map(target_values)

Transform_cat_data = pd.DataFrame()
for data in Tcat_data:
    Transform_cat_data[data] = le.fit_transform(Tcat_data[data])
X = pd.concat([Tnum_data,Transform_cat_data],axis=1)
y = target
## Spliting the dataset for training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
## Training and testing the dataset
model.fit(X_train,y_train)    
y_pred = model.predict(X_test)

## Calculating accuracy score
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
print(acc)

## Saving moodel
joblib.dump(model,'model.sav')


