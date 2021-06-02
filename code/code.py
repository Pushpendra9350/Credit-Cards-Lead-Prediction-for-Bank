"""
@Author: Pushpendra Kumar
For more analysis go to .ipynb
This file is only containes final solution code
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# Simply to read the data from csv file.
data = pd.read_csv("train.csv")

# To check the shape of the dataframe we have
print("Shape of data is: ",data.shape)
print("Number of rows we have: ",data.shape[0])
print("Number of columns we have: ",data.shape[1])

# Lets drop duplicates
data.drop_duplicates(subset=['Gender','Age','Region_Code','Occupation', 'Channel_Code','Vintage','Credit_Product','Avg_Account_Balance','Is_Active','Is_Lead'],keep="first",inplace=True)

# This is like onehot encoding 
# Apply one hot encoding based on the occupation 
data = pd.concat([data,pd.get_dummies(data.Occupation)],axis=1)
# Then we have to drop the occupation column from the main data
data.drop(["Occupation"],axis=1,inplace=True)

# Apply one hot encoding based on the region code 
data = pd.concat([data,pd.get_dummies(data.Region_Code)],axis=1)
data.drop(["Region_Code"],axis=1,inplace=True)

data = pd.concat([data,pd.get_dummies(data.Channel_Code)],axis=1)
data.drop(["Channel_Code"],axis=1,inplace=True)


# To encode Is_Active in 0 and 1 with map function 
data["Is_Active"] = data["Is_Active"].map({"No":0,"Yes":1})

# To encode Gender in numerical data 
data["Gender"] = data["Gender"].map({"Female":0,"Male":1})

data = pd.concat([data,pd.get_dummies(data.Credit_Product,prefix='Credit_Product')],axis=1)
data.drop("Credit_Product",inplace=True,axis=1)

# Select to 20 features
ndata = data[["Age","Avg_Account_Balance","Vintage","Credit_Product_No","Is_Active","X1","X2","X3","Entrepreneur","Credit_Product_Yes","Salaried","Self_Employed","RG268","RG283","Gender","RG284","RG270","RG252","RG261","RG264"]]
y = data["Is_Lead"]

# Training model
model_lgbm = LGBMClassifier()
model_lgbm.fit(ndata, y,verbose=False)

# Read the test file for predictions
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/newone/test.csv")

# This is like onehot encoding 
# Apply one hot encoding based on the occupation 
test = pd.concat([test,pd.get_dummies(test.Occupation)],axis=1)
# Then we have to drop the occupation column from the main data
test.drop(["Occupation"],axis=1,inplace=True)

# Apply one hot encoding based on the region code 
test = pd.concat([test,pd.get_dummies(test.Region_Code)],axis=1)
test.drop(["Region_Code"],axis=1,inplace=True)

test = pd.concat([test,pd.get_dummies(test.Channel_Code)],axis=1)
test.drop(["Channel_Code"],axis=1,inplace=True)


# To encode Is_Active in 0 and 1 with map function 
test["Is_Active"] = test["Is_Active"].map({"No":0,"Yes":1})

# To encode Gender in numerical data 
test["Gender"] = test["Gender"].map({"Female":0,"Male":1})

test = pd.concat([test,pd.get_dummies(test.Credit_Product,prefix='Credit_Product')],axis=1)
test.drop("Credit_Product",inplace=True,axis=1)

# Selcting same important features
ntest = test[["ID","Age","Avg_Account_Balance","Vintage","Credit_Product_No","Is_Active","X1","X2","X3","Entrepreneur","Credit_Product_Yes","Salaried","Self_Employed","RG268","RG283","Gender","RG284","RG270","RG252","RG261","RG264"]]

# To get Test data id's
ids = ntest["ID"].to_list()

# Now drop id's from test data
ntest.drop("ID",axis=1,inplace=True)

# Lets predict the result on test data
testpred = model_lgbm.predict_proba(ntest)

# get probablities values 
val = pd.Series(testpred[:,1]).to_list()

# Make a dataframe to submit 
res = pd.DataFrame({"ID":ids,"Is_Lead":val})

# Save this dataframe for submission
res.to_csv("xgb_20f_submissionV13.csv",index=False)