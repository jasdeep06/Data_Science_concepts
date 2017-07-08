import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("dataset/tc_house_price.csv")


missing_lot=(data.loc[data["LotSize"].isnull()]).index
data.drop(missing_lot, axis=0, inplace=True)
missing_bath = (data.loc[data["BATHS"].isnull()]).index
data.drop(missing_bath, axis=0, inplace=True)


numeric_features=data.select_dtypes(include=[np.number])
for feature in numeric_features:
    data[feature+"Squared"]=data[feature]**2
    data[feature+"Centered"]=data[feature]-(data[feature].mean())
    data[feature+"CenteredSquared"]=data[feature+"Centered"]**2
    print(feature+","+ feature +"Squared",data[feature].corr(data[feature+"Squared"]))
    print(feature + "Centered," + feature + "CenteredSquared", data[feature+"Centered"].corr(data[feature + "CenteredSquared"]))




plt.scatter(data["DOMCentered"],data["DOMCenteredSquared"],label="DOM")
#plt.hist(data["DOMCenteredSquared"])
#plt.hist(data["BEDSCentered"])
plt.show()
print(data["LotSize"].var())
data["LotSizeSquared"]=data["LotSize"]**2
data["LotSizeCentered"]=data["LotSize"]-(data["LotSize"].mean())
data["LotSizeCenteredSquared"]=data["LotSizeCentered"]**2

print(data["LotSize"].corr(data["LotSizeSquared"]))
print(data["LotSizeCentered"].corr(data["LotSizeCenteredSquared"]))