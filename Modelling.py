# -*- coding: utf-8 -*-
"""

@author: Harpreet Singh
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression  ,BayesianRidge,HuberRegressor,ARDRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score,r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


from lightgbm import LGBMRegressor as lgr
import lightgbm as lgb
from catboost import CatBoostRegressor as cbr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso,LinearRegression,Ridge,ElasticNet
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,RobustScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV , StratifiedKFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler ,PolynomialFeatures,minmax_scale,MaxAbsScaler ,LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR
from xgboost import XGBRegressor





wind_data = pd.read_csv('C:\\Users\\Harpreet Singh\\Documents\\Machine Learning\\Untitled Folder\\wind_data.csv')

wind_data['date_column'] = pd.to_datetime(wind_data['date_column'])

wind_data["Year"]=wind_data["date_column"].dt.year

wind_data["Month"]=wind_data["date_column"].dt.month
wind_data["Day"]=wind_data["date_column"].dt.day

wind_data = wind_data.set_index(['date_column'])

X = wind_data.drop(columns="wind_generation_actual")

y = wind_data["wind_generation_actual"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Size of X Train : ", X_train.shape)
print("Size of X Test  : ", X_test.shape)
print("Size of Y Train : ", y_train.shape)
print("Size of Y Test  : ", y_test.shape)





#1. Linear Model

LinearRegression_model=LinearRegression(fit_intercept=True,normalize=False,copy_X=True, n_jobs=None)

LinearRegression_model.fit(X_train,y_train)

print("Score the X-train with Y-train is : ", LinearRegression_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", LinearRegression_model.score(X_test,y_test))

# Predictions 
y_pred_LR=LinearRegression_model.predict(X_test)

#Model Evaluation 
print( " Model Evaluation Linear R : mean absolute error is ", mean_absolute_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : mean squared  error is " , mean_squared_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : median absolute error is " ,median_absolute_error(y_test,y_pred_LR)) 






# 2. Model: ARD Regression
ARDRegression_model=ARDRegression(n_iter=30,tol=4,alpha_1=1e-06,alpha_2=1e-06,lambda_1=1e-06,lambda_2=1e-06,compute_score=False,threshold_lambda=10000.0,fit_intercept=True,normalize=False,copy_X=True,verbose=False)

# fit model

ARDRegression_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", ARDRegression_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", ARDRegression_model.score(X_test,y_test))

# Expected value Y using X test
y_predARD=ARDRegression_model.predict(X_test)

# Model Evaluation
print( " Model Evaluation Linear R : mean absolute error is ", mean_absolute_error(y_test,y_predARD))
print(" Model Evaluation Linear R : mean squared  error is " , mean_squared_error(y_test,y_predARD))
print(" Model Evaluation Linear R : median absolute error is " ,median_absolute_error(y_test,y_predARD)) 










# 3. Decision Tree Regressor Model

DecisionTreeRegressor_model=DecisionTreeRegressor(splitter='best',max_depth=100)

# fit model

DecisionTreeRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", DecisionTreeRegressor_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", DecisionTreeRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predDTR=DecisionTreeRegressor_model.predict(X_test)








#4. K Neighbors Regressor 


KNeighborsRegressor_model=KNeighborsRegressor(n_neighbors=6,weights='uniform',algorithm='auto',leaf_size=20,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None)

# fit model

KNeighborsRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", DecisionTreeRegressor_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", DecisionTreeRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predKN=KNeighborsRegressor_model.predict(X_test)








#5. Random Forest Regressor Model
RandomForestRegressor_model=RandomForestRegressor(n_estimators=100,ccp_alpha=20)

# fit model

RandomForestRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", RandomForestRegressor_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", RandomForestRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predRFR=RandomForestRegressor_model.predict(X_test)









#6. MLP Regressor

MLPRegressor_model=MLPRegressor()

# fit model

MLPRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", MLPRegressor_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", MLPRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predMLP=MLPRegressor_model.predict(X_test)









#7. SVR Regressor Model

svr_model=SVR(degree=1)

# fit model

svr_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", svr_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", svr_model.score(X_test,y_test))

# Expected value Y using X test
y_predsvr=svr_model.predict(X_test)



#8. Lasso
lasso = Lasso(alpha =0.0005, random_state=20)
param_grid = [{'alpha':[0.0005,0.001, 0.005, 0.01, 0.05, 0.03, 0.1, 0.5, 1]}]
lasso_model = lasso.fit(X_train,y_train)
y_lasso=lasso_model.predict(X_test)






#9. Catboost 
cat = cbr(loss_function='RMSE',learning_rate=0.01,max_depth=7,iterations=1500) 
catreg = cat.fit(X_train,y_train,verbose_eval=200,plot=True,eval_set=(X_test, y_test))




#Plots 
y_test_temp = y_test.copy() 
y_test_temp.to_frame()

 
predictions=LinearRegression_model.predict(X_test)
Preds = pd.DataFrame(predictions) 

predictions= lasso_model.predict(X_test)
Preds['lasso_prediction'] = predictions

predictions= catreg.predict(X_test)
Preds['cat_boost_prediction']=predictions

predictions= catreg.predict(X_test)
Preds['Linear']=y_pred_LR


predictions= catreg.predict(X_test)
Preds['Decision Tree']=y_predDTR


predictions= catreg.predict(X_test)
Preds['K-Nearest Means']=y_predKN


predictions= catreg.predict(X_test)
Preds['Random Forest']=y_predRFR


predictions= catreg.predict(X_test)
Preds['ARD']=y_predARD


predictions= catreg.predict(X_test)
Preds['SVR']=y_predsvr


predictions= catreg.predict(X_test)
Preds['MLP']=y_predMLP


#Concatinating everything 
tif = pd.concat([Preds, y_test_temp.reset_index()], axis=1)
tif.set_index('date_column')


fig,ax = plt.subplots(ncols=1,nrows=9,figsize=(25,7))
tif.set_index('date_column')[['Linear','wind_generation_actual']].plot(ax=ax[0])
tif.set_index('date_column')[['lasso_prediction','wind_generation_actual']].plot(ax=ax[1])
tif.set_index('date_column')[['cat_boost_prediction','wind_generation_actual']].plot(ax=ax[2])
tif.set_index('date_column')[['Decision Tree','wind_generation_actual']].plot(ax=ax[3])
tif.set_index('date_column')[['K-Nearest Means','wind_generation_actual']].plot(ax=ax[4])
tif.set_index('date_column')[['Random Forest','wind_generation_actual']].plot(ax=ax[5])
tif.set_index('date_column')[['ARD','wind_generation_actual']].plot(ax=ax[6])
tif.set_index('date_column')[['SVR','wind_generation_actual']].plot(ax=ax[7])
tif.set_index('date_column')[['MLP','wind_generation_actual']].plot(ax=ax[8])

plt.xlabel('Power (MW)')
plt.show()

#Model Evaluation 
print(" R2 score of Linear Regression " ,r2_score(y_test,y_pred_LR)) 
print(" R2 score of ARD Regression " ,r2_score(y_test,y_predARD)) 
print(" R2 score of Decision Tree Regressor model " ,r2_score(y_test,y_predDTR)) 
print(" R2 score of K Neighbors Regressor Model " ,r2_score(y_test,y_predKN)) 
print(" R2 score of Random Forest Regressor Model " ,r2_score(y_test,y_predRFR)) 
print(" R2 score of MLP Regressor Model " ,r2_score(y_test,y_predMLP)) 
print(" R2 score of SVR  Model " ,r2_score(y_test,y_predsvr))
print(" R2 score of Lasso  Model " ,r2_score(y_test,y_lasso))
print(" R2 score of Catboost  Model " ,r2_score(y_test,y_predsvr))


#Next steps, merge weather data to increase the R2 score of the models. Best model among all seems to be random forest regressor 
