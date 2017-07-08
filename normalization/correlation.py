import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading data
data=pd.read_csv("dataset/tc_house_price.csv")

#missing data removal
missing_lot=(data.loc[data["LotSize"].isnull()]).index
data.drop(missing_lot, axis=0, inplace=True)
missing_bath = (data.loc[data["BATHS"].isnull()]).index
data.drop(missing_bath, axis=0, inplace=True)

#numeric features for centering
numeric_features=data.select_dtypes(include=[np.number])

#squared features centering
for feature in numeric_features:
    data[feature+"Squared"]=data[feature]**2
    data[feature+"Centered"]=data[feature]-(data[feature].mean())
    data[feature+"CenteredSquared"]=data[feature+"Centered"]**2
    
    #printing correlations between (feature,squared feature) and (centered feature,centered squared feature)
    print(feature+","+ feature +"Squared",data[feature].corr(data[feature+"Squared"]))
    print(feature + "Centered," + feature + "CenteredSquared", data[feature+"Centered"].corr(data[feature + "CenteredSquared"]))




