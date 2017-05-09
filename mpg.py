#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:33:54 2017

@author: stephanosarampatzes
"""

import pandas as pd
import numpy as np

# load the dataset
data=pd.read_csv('auto-mpg-nameless.csv')

#Correlation matrix
corr_matrix=data.corr().sort_values(ascending=False,by='mpg')

# correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix,annot=True,vmax=0.8,square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('features correlated to the target value')
plt.tight_layout()
plt.show()

# weight and mpg linear plot
f, ax = plt.subplots(figsize=(15, 6))
plt.subplot(1,2,1)
plt.scatter(data.weight,data.mpg)
plt.xlabel('weight',fontsize=13)
plt.ylabel('mpg',fontsize=13)
plt.subplot(1,2,2)
sns.regplot(data.weight,data.mpg)
plt.show()

# plots in descending order of correlation
from pylab import *

f, ax = plt.subplots(figsize=(15, 10))
plt.suptitle('numerical-continuous variables',fontsize=16)

plt.subplot(2,2,1)
plt.scatter(data.weight,data.mpg)
plt.xlabel('weight')
plt.ylabel('mpg')

plt.subplot(2,2,2)
plt.scatter(data.displacement,data.mpg);
plt.xlabel('displacement')
plt.ylabel('mpg')

plt.subplot(2,2,3)
plt.scatter(data.hp,data.mpg);
plt.xlabel('HP')
plt.ylabel('mpg')

plt.subplot(2,2,4)
plt.scatter(data.acc,data.mpg)
plt.xlabel('ACC')
plt.ylabel('mpg')
plt.tight_layout
plt.show()

# discrete numerical plot
f, ax = plt.subplots(figsize=(12, 7))
plt.suptitle('mpg over the years')
plt.subplot(2,1,1)
sns.barplot(data.year, data.mpg)
plt.subplot(2,1,2)
sns.barplot(data.year,data.mpg,hue=data.origin)
plt.tight_layout()
plt.show()

# 1.US , 2.European , 3. Japanese
plt.suptitle('mpg per origin')
sns.barplot(x='origin',y='mpg',data=data)
plt.show()

# Group the 'mpg' and other features by origin
grouped_mean=data.groupby(['origin']).mean()
grouped_mean.drop(grouped_mean.columns[6],axis=1,inplace=True)


# Convert Cubic Inches (cin) to Cubic Centimeters (cc)
data['cc']=np.array(data.displacement)*16.38706
print(data.head(2))

# looking for NA's
data.isnull().sum()

# outliers
plt.scatter(data.displacement,data.mpg);
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.show()

# a stripplot to identify the origin of outlier(?)
sns.stripplot(x='displacement',y='mpg',hue='origin',data=data)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# locate outlier's index
print(data.loc[(data['displacement'] > 250) & (data['mpg']>35)])

# normality of distributions
from scipy import stats
from scipy.stats import norm

# mpg
sns.distplot(data.mpg,hist=True,fit=norm);
fig=plt.figure()
prob_plot=stats.probplot(data['mpg'],plot=plt)
plt.show()
print('mpg skewness:',data.mpg.skew())

# weight
sns.distplot(data.weight,hist=True,fit=norm);
fig=plt.figure()
prob_plot=stats.probplot(data['weight'],plot=plt)
plt.show()
print('weight skewness:',data.weight.skew())

# displacement
sns.distplot(data.displacement,hist=True,fit=norm);
fig=plt.figure()
prob_plot=stats.probplot(data.displacement,plot=plt)
plt.show()
print('disp skewness:',data.displacement.skew())

# hp
sns.distplot(data.hp,hist=True,fit=norm);
fig=plt.figure()
prob_plot=stats.probplot(data['hp'],plot=plt)
plt.show()
print('hp skewness:',data.hp.skew())

# log transformation

data['mpg']=np.log(data.mpg)
data['log_wg']=np.log(data.weight)
data['log_cc']=np.log(data.cc)
data['log_hp']=np.log(data.hp)
data['log_dp']=np.log(data.displacement)

# Create new features
# 3* Polynomials on the top 5 correlated features

data['weight2']=data['weight']**2
data['weight3']=data['weight']**3
data['weightsq']=np.sqrt(data['weight'])

data['cc2']=data['cc']**2
data['cc3']=data['cc']**3
data['ccsq']=np.sqrt(data['cc'])


data['displacement2']=data['displacement']**2
data['displacement3']=data['displacement']**3
data['displacementsq']=np.sqrt(data['displacement'])

data['displacement2']=data['displacement']**2
data['displacement3']=data['displacement']**3
data['displacementsq']=np.sqrt(data['displacement'])

data['cyls2']=data['cyls']**2
data['cyls3']=data['cyls']**3
data['cylssq']=np.sqrt(data['cyls'])

data['hp2']=data['hp']**2
data['hp3']=data['hp']**3
data['hpsq']=np.sqrt(data['hp'])

# building our 5 models
from sklearn import model_selection,cross_validation
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,LassoLars,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

# cross validation

X=data.iloc[:,1:29]
y=data.iloc[:,0]

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=45)

print('linear regression')
lnr = LinearRegression()
lnr.fit(X_train,y_train)
pred_1 = lnr.predict(X_test)
print(round(r2_score(y_test,pred_1)*100,2),'%')
print(np.sqrt(mean_squared_error(y_test,pred_1)))
print()

print('random forest')
rf=RandomForestRegressor(n_estimators=100, oob_score = True,random_state =45)
rf.fit(X_train,y_train)
pred_2 = rf.predict(X_test)
print(round(r2_score(y_test,pred_2)*100,2),'%')
print(np.sqrt(mean_squared_error(y_test,pred_2)))
print()

print('ridge')
ridge = Ridge(alpha = 1)
rd=ridge.fit(X_train, y_train)
pred_ridge=rd.predict(X_test)
print(round(r2_score(y_test,pred_ridge)*100,2),'%')
print(np.sqrt(mean_squared_error(y_test,pred_ridge)))
print()

print('lasso')
laso=Lasso(alpha = 0.00001)
laso=laso.fit(X_train,y_train)
pred_laso=laso.predict(X_test)
print(round(r2_score(y_test,pred_laso)*100,2),'%')
print(np.sqrt(mean_squared_error(y_test,pred_laso)))
print()

print('Elastic Net')
EN=ElasticNet(alpha=0.00001)
EN=EN.fit(X_train,y_train)
pred_EN=EN.predict(X_test)
print(round(r2_score(y_test,pred_EN)*100,2),'%')
print(np.sqrt(mean_squared_error(y_test,pred_EN)))

# feature importance plot for ...

# Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances in the RF model")
plt.bar(range(X_train.shape[1]), importances,color="r",align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout
plt.xticks(rotation=90)
plt.show()

# Lasso
coefs = pd.Series(laso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()
