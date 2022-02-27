import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
import lightgbm as lgb

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from scipy import stats
from sklearn import linear_model
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostRegressor

data = pd.read_csv(r"house.txt")
print(data.shape)
data.info()

def plotting(p):
    plt.figure(figsize = (15,6))
    sns.histplot(p,palette='Blues_r',kde=True)
    plt.axvline(x=p.mean(), color='g', linestyle='--', linewidth=3)
    plt.text(p.mean(), 125, "Mean", horizontalalignment='left', size=20, color='black', weight='semibold')
    plt.title('Sale Price Distribution Before Training',fontsize=20)
    plt.show()
    
def plotting2(p):
    plt.figure(figsize = (15,6))
    sns.histplot(p,palette='Blues_r',kde=True)
    plt.axvline(x=p.mean(), color='g', linestyle='--', linewidth=3)
    plt.text(p.mean(), 125, "Mean", horizontalalignment='left', size=20, color='black', weight='semibold')
    plt.title('Sale Price Distribution After Training',fontsize=20)
    plt.show()    
    
plotting(p = data['SalePrice'])

dropNanByDensity = lambda dataFrame, density: [ key for key in dataFrame if (dataFrame[key].isna().sum() / dataFrame.shape[0]) > density ] 
# drop according feature depended to null density, Ex. dropNanByDensity(data, 0.1)
getDummyColumns = lambda dataFrame : [ key for key in dataFrame if data[key].dtype == 'O' ] 
# get features that will be processed for integer encoding by targeting Object type features
getNumericColumns = lambda dataFrame : [ key for key in dataFrame if data[key].dtype != 'O' ] 


a = data[dropNanByDensity(data,0.005)].dropna() #drop null features columns
data = data.drop(a, axis=1)
#data = data.drop(columns= ["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis=1)


clf = IsolationForest(  max_samples="auto", random_state = 1, contamination= 'auto')
preds = clf.fit_predict(data[getNumericColumns(data)].dropna())

data = data.drop(labels=np.where(preds == -1)[0], axis=0) #-1 --> anomaly, axis=0 row


data = data.interpolate(method='spline', order=2)
#test e de fit ediyo gibimsi rezalet daha sonra bakılır ayırmaya 

data = pd.get_dummies(data, prefix = getDummyColumns(data))

def scatter(df):
    [df.plot(x = key, y = "SalePrice", kind = "scatter") for key in df.keys()]
   
def LGBM():
    model=lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.035, n_estimators=2177, max_bin=50, bagging_fraction=0.65,bagging_freq=5, bagging_seed=7, 
                                feature_fraction=0.201, feature_fraction_seed=7,n_jobs=-1)
    print("The model is LGBM")                            
    return model

def ExtraTreesRegressor():
    model = ExtraTreesRegressor(n_estimators=100, random_state=0)
    print("The model is Extra Tree Regressor")
    return model

def XGB():
    model = XGBRegressor()
    print("The model is XGB")
    return model

def GBR():
    model = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03005, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=14, loss='huber', random_state =42)
    print("The model is  GBR")
    return model


def RandomForest():
    model = RandomForestRegressor(n_estimators=1000)
    print("The model is Random Forest Regression")
    return model

def LinearReg():
    model = linear_model.LinearRegression()
    print("The model is Linear Regression")
    return model 

def SVM():
    model = SVR(kernel='rbf', C=1000000, epsilon=0.001)
    print("The model is SVM")
    return model
    
def Ridge():
    model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
    print("The model is Ridge ")
    return model
    
def Lasso():
    model = Lasso(alpha=0.1, precompute=True, positive=True, selection='random',random_state=42)
    print("The model is Lasso")
    return model

def ElasticNet():
    model =  ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
    print("The model is Elastic Net")
    return model

model = LGBM()


def train(model):

    X, y = shuffle(data.drop(columns=['SalePrice']).values, data.SalePrice.values, random_state=13)
    #X = data.drop(columns=['SalePrice']).values
    #y = data.SalePrice.values
    X = X.astype(np.float32)
    all_X = X
    all_y = y
    X_train, X_test, y_train, y_test = train_test_split(
        all_X, all_y, test_size=0.1,random_state=0)
    corr_plot(d = data)
#    feature_corr_plt(n =data.Neighborhood , l = data.LotFrontage, d = data)
    model.fit(X_train, y_train)
    predict(X_train,X_test, y_test)
    cross_val(X, y)   
    
    
def cross_val(X, y):
    scores = cross_val_score(model, X, y, cv=10) #all 
    print("Accuracy: ",scores.mean())


def print_evaluate(actual, predicted):  
    mae = metrics.mean_absolute_error(actual, predicted)
    mse = metrics.mean_squared_error(actual, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(actual, predicted))
    r2_square = metrics.r2_score(actual, predicted)
    print('Test set evaluation:\n_____________________________________')
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(actual, predicted):
    mae = metrics.mean_absolute_error(actual, predicted)
    mse = metrics.mean_squared_error(actual, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(actual, predicted))
    r2_square = metrics.r2_score(actual, predicted)
    return mae, mse, rmse, r2_square


def predict(X_train, X_test, y_test): 
    test_pred=model.predict(X_train)
    predictions = model.predict(X_test)
    print(predictions)
    print_evaluate(y_test, predictions)
    plotFunc(y_test,predictions)


def plotFunc(y_test,predictions):
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, predictions, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    
    p1 = max(max(predictions), max(y_test))
    p2 = min(min(predictions), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()



def corr_plot(d):
    corr=d.corr().abs()
    n_most_correlated=12
    most_correlated_feature=corr['SalePrice'].sort_values(ascending=False)[:n_most_correlated].drop('SalePrice')
    most_correlated_feature_name=most_correlated_feature.index.values
    f, ax = plt.subplots(figsize=(15, 6))
    plt.xticks(rotation='90')
    sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)
    plt.title("The Most Correlated Values with Sale Price")
    plt.show()

train(model)
plotting2(p = data['SalePrice'])

