"""
-------------------------------------------------------
CP468 Project
-------------------------------------------------------
Author:  Zehao Liu/ Jialong Zhang
ID:      193074000/190227130
Email:  liux4000@mylaurier.ca/ zhan2713@mylaurier.ca
Github： https://github.com/liuzehaoleon/468project
__updated__ = '2022-07-19'
-------------------------------------------------------
"""


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
sns.set()


URL="https://raw.githubusercontent.com/liuzehaoleon/468project/main/train-data.csv"
train_data = pd.read_csv(URL)

""" train_data.info()
print(train_data.head())
print(train_data.tail()) """
train_data = train_data.iloc[:,1:]

""" print(train_data.describe())
print(train_data.shape) """
#delete the records with NULL values
""" print(train_data.isnull().sum()) """
train_data.drop(["New_Price"],axis=1,inplace=True)
train_data = train_data.dropna(how='any')
train_data = train_data.reset_index(drop=True)


#Change the 'Name' to 'Cars'
train_data['Cars'] = train_data['Name'].str.split(" ").str[0] + ' ' +train_data['Name'].str.split(" ").str[1]
""" print(train_data.head()) """

# remove some units from the data
train_data['Mileage'] = train_data['Mileage'].str.replace(' kmpl','')
train_data['Mileage'] = train_data['Mileage'].str.replace(' km/kg','')
train_data['Engine'] = train_data['Engine'].str.replace(' CC','')
train_data['Power'] = train_data['Power'].str.replace('null bhp','112')
train_data['Power'] = train_data['Power'].str.replace(' bhp','')

train_data['Mileage'] = train_data['Mileage'].astype(float)
train_data['Engine'] = train_data['Engine'].astype(float)
train_data['Power'] = train_data['Power'].astype(float)
""" print(train_data.shape) """

feature = ['Cars', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 
           'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats','Price']
data = pd.DataFrame(train_data, columns=feature)
""" print(data.head())
print(train_data.isnull().sum()) """

#Data Visualization
#distribution of prices
plt.figure(figsize=(10,10))
sns.distplot(data['Price'])
plt.show()

#applying log transformation
data['Price'] = np.log(data['Price'])
#sns.distplot(data['Price']);
plt.figure(figsize=(10,10))
sns.distplot(data['Price'], fit=None)
plt.show()


#heatmap
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn')
plt.show()

#box
plt.figure(figsize=(15,10))
xl = 'Year'
yl = 'Price'
sns.boxplot(data=data, x=xl, y=yl, hue='Transmission')
plt.xlabel('{}'.format(xl), size=14)
plt.ylabel('{}'.format(yl), size=14)
plt.show()

plt.figure(figsize=(15,10))
xl = 'Year'
yl = 'Price'
sns.boxplot(data=data, x=xl, y=yl, hue='Fuel_Type')
plt.xlabel('{}'.format(xl), size=14)
plt.ylabel('{}'.format(yl), size=14)
plt.show()

#Handling Categorical parameters
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder().fit(data['Cars'])
data['Cars'] = label_encoder.transform(data['Cars'])

label_encoder = LabelEncoder().fit(data['Location'])
data['Location'] = label_encoder.transform(data['Location'])

label_encoder = LabelEncoder().fit(data['Fuel_Type'])
data['Fuel_Type'] = label_encoder.transform(data['Fuel_Type'])

label_encoder = LabelEncoder().fit(data['Transmission'])
data['Transmission'] = label_encoder.transform(data['Transmission'])

label_encoder = LabelEncoder().fit(data['Owner_Type'])
data['Owner_Type'] = label_encoder.transform(data['Owner_Type'])

""" print(data.head()) """

#Define Date
def Definedata():
    fdata = data[feature]
    X = fdata.drop(columns=['Price']).values
    y0 = fdata['Price'].values
    lab_enc = preprocessing.LabelEncoder()
    y = lab_enc.fit_transform(y0)
    return X, y


#Build Model
models = [['DecisionTreeRegressor', DecisionTreeRegressor()],
          ['RandomForestRegressor', RandomForestRegressor()],
          ['LinearRegression', LinearRegression()],
          ['KNeighborsClassifier', KNeighborsClassifier()],
          ['DecisionTreeClassifier', DecisionTreeClassifier()],
          ['Neural Network', MLPClassifier()]]

#Training
X, y = Definedata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 25)
model_details = []
for mod in models:
    modle_d_list = []
    name = mod[0]
    model = mod[1]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    TrS =  model.score(X_train,y_train)
    TeS = model.score(X_test,y_test)
    modle_d_list.append(name)
    modle_d_list.append(RMSE)
    modle_d_list.append(TrS)
    modle_d_list.append(TeS)
    model_details.append(modle_d_list)

df = pd.DataFrame(model_details, columns=['model','Root Mean Squared  Error','Accuracy on Traing set','Accuracy on Testing set'])
print(df.sort_values(by='Accuracy on Testing set'))
#final Error Table
def Models(models):
    model = models
    X, y = Definedata()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 25)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print("\t\tError Table")
    print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Accuracy on Traing set   : ', model.score(X_train,y_train))
    print('Accuracy on Testing set  : ', model.score(X_test,y_test))

Models(RandomForestRegressor(n_estimators=10000, max_features='sqrt',max_depth=25))








