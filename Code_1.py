import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv('train.csv')
X=data.iloc[:,2:].values
y=data.iloc[:,1].values
import xgboost
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)



model=xgboost.XGBRegressor()
model.fit(X_train,y_train)

y_predict=model.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.metrics import r2_score

score=r2_score(y_test,y_predict)
