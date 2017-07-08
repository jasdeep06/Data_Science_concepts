import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def dataset_generation():
    
    #reading_data
    data=pd.read_csv("dataset/tc_house_price.csv")
    
    #missing values in LotSize and BATHS
    missing_lot=(data.loc[data["LotSize"].isnull()]).index
    missing_bath=(data.loc[data["BATHS"].isnull()]).index
    #dropping missing values
    data.drop(missing_lot,axis=0,inplace=True)
    data.drop(missing_bath,axis=0,inplace=True)
    
    #dropping variables latitude and longitude to simplify analysis
    data.drop("LATITUDE",axis=1,inplace=True)
    data.drop("LONGITUDE",axis=1,inplace=True)

    
    
    #Standardizing numeric data(for generating unstandardized file comment this block)
    numeric_features=list(data.select_dtypes(include=[np.number]))
    numeric_features.remove("ListPrice")
    scaler=StandardScaler().fit(data[numeric_features])
    data[numeric_features]=scaler.transform(data[numeric_features])

    #Adding self interaction terms(square,cube,quart)
    for feature in numeric_features:
        data["Squared"+feature]=data[feature]**2
        data["cubed"+feature]=data[feature]**3
        data["quadra"+feature]=data[feature]**4

    
    
    #converting binary string variables to 0,1
    data.BeenReduced[data.BeenReduced=="Y"]=1
    data.BeenReduced[data.BeenReduced=="N"]=0
    data.SoldPrev[data.SoldPrev=="Y"]=1
    data.SoldPrev[data.SoldPrev=="N"]=0
    data.ShortSale[data.ShortSale=="Y"]=1
    data.ShortSale[data.ShortSale=="N"]=0
    data.HasGarage[data.HasGarage=="Garage"]=1
    data.HasGarage[data.HasGarage=="No"]=0


    #splitting into test,train data
    train_data=data.iloc[0:700]
    test_data=data.iloc[700:]
    train_data.to_csv("dataset/quadratic_standardized_train.csv")
    test_data.to_csv("dataset/quadratic_standardized_test.csv")

#generating data.After generation in subsequent run comment this.
dataset_generation()


#reading X data
X_train1=pd.read_csv("dataset/quadratic_standardized_train.csv")
X_test1=pd.read_csv("dataset/quadratic_standardized_test.csv")

#Y data
y_train1=X_train1["ListPrice"]
y_test1=X_test1["ListPrice"]

#dropping listprice of X data
X_train1.drop("ListPrice",axis=1,inplace=True)
X_test1.drop("ListPrice",axis=1,inplace=True)

#linear regression
model1=LinearRegression()
model1.fit(X_train1,y_train1)
prediction1=model1.predict(X_test1)

#prediction rmse
error1=np.sqrt(np.mean((prediction1-y_test1)*(prediction1-y_test1)))

#rmse
print(error1)
#R^2 value
print(r2_score(y_true=y_test1,y_pred=prediction1))
#coefficient
print(model1.coef_)


#same for non standardized
X_train2=pd.read_csv("dataset/quadratic_train.csv")
X_test2=pd.read_csv("dataset/quadratic_test.csv")

y_train2=X_train2["ListPrice"]
y_test2=X_test2["ListPrice"]

X_train2.drop("ListPrice",axis=1,inplace=True)
X_test2.drop("ListPrice",axis=1,inplace=True)
print("correlation ",X_train2["LotSize"].corr(X_train2["quadraLotSize"]))

model2=LinearRegression()
model2.fit(X_train2,y_train2)
prediction2=model2.predict(X_test2)
error2=np.sqrt(np.mean((prediction2-y_test2)*(prediction2-y_test2)))
print(error2)
print(r2_score(y_true=y_test2,y_pred=prediction2))

print(model2.coef_)
