import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold

data_input = pd.read_csv('OER-database-noDOI.csv', sep=',')

labels = data_input['Tafel'] # can change to "Overpotential"
features = data_input.drop('Tafel', axis=1) # can change to "Overpotential"

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

xgbr_model = XGBRegressor()
xgbr_model.fit(X_train, y_train)

rfr_model = RandomForestRegressor()
rfr_model.fit(X_train, y_train)

etr_model = ExtraTreesRegressor()
etr_model.fit(X_train, y_train)

knnr_model = KNeighborsRegressor()
knnr_model.fit(X_train, y_train)

dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_train, y_train)

svr_model = SVR()
svr_model.fit(X_train, y_train)

gbr_model = GradientBoostingRegressor()
gbr_model.fit(X_train, y_train)

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

'Linear': cross_val_score(lr_model, features, labels, scoring='r2', cv=kfold).mean(),
'XGBoost': cross_val_score(xgbr_model, features, labels, scoring='r2', cv=kfold).mean(),
'RT': cross_val_score(rfr_model, features, labels, scoring='r2', cv=kfold).mean(),
'ET': cross_val_score(etr_model, features, labels, scoring='r2', cv=kfold).mean(),
'KNN': cross_val_score(knnr_model, features, labels, scoring='r2', cv=kfold).mean(),
'DT': cross_val_score(dtr_model, features, labels, scoring='r2', cv=kfold).mean(),
'SVR': cross_val_score(svr_model, features, labels, scoring='r2', cv=kfold).mean(),
'GBR': cross_val_score(gbr_model, features, labels, scoring='r2', cv=kfold).mean()
