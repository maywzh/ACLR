import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None)
df.head()
df.tail()

##Defining headers for the DataSet
headers=["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels",
        "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
         "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
         "highway-mpg", "price"]


df.columns = headers
df=df.replace('alfa-romero',1)
df=df.replace('audi',2)
df=df.replace('bmw',3)
df=df.replace('chevrolet',4)
df=df.replace('dodge',5)
df=df.replace('honda',6)
df=df.replace('isuzu',7)
df=df.replace('jaguar',8)
df=df.replace('mazda',9)
df=df.replace('mercedes-benz',10)
df=df.replace('mercury',11)
df=df.replace('mitsubishi',12)
df=df.replace('nissan',13)
df=df.replace('peugot',14)
df=df.replace('plymouth',15)
df=df.replace('porsche',16)
df=df.replace('renault',17)
df=df.replace('saab',18)
df=df.replace('subaru',19)
df=df.replace('toyota',20)
df=df.replace('volkswagen',21)
df=df.replace('volvo',22)

df["aspiration"].replace('std', 0, inplace = True)
df["aspiration"].replace('turbo', 1, inplace = True)


df["num-of-doors"].replace('two', 0, inplace = True)
df["num-of-doors"].replace('four', 1, inplace = True)
df["num-of-doors"].replace('?', 1, inplace = True)


df["body-style"].replace('convertible', 0, inplace = True)
df["body-style"].replace('hatchback', 1, inplace = True)
df["body-style"].replace('hardtop', 2, inplace = True)
df["body-style"].replace('sedan', 3, inplace = True)
df["body-style"].replace('wagon', 4, inplace = True)



df["drive-wheels"].replace('rwd', 0, inplace = True)
df["drive-wheels"].replace('fwd', 1, inplace = True)
df["drive-wheels"].replace('4wd', 2, inplace = True)



df["engine-type"].replace('dohc', 0, inplace = True)
df["engine-type"].replace('dohcv', 1, inplace = True)
df["engine-type"].replace('ohc', 2, inplace = True)
df["engine-type"].replace('ohcf', 3, inplace = True)
df["engine-type"].replace('ohcv', 4, inplace = True)
df["engine-type"].replace('rotor', 5, inplace = True)
df["engine-type"].replace('l', 6, inplace = True)


df["num-of-cylinders"].replace('two', 2, inplace = True)
df["num-of-cylinders"].replace('three', 3, inplace = True)
df["num-of-cylinders"].replace('four', 4, inplace = True)
df["num-of-cylinders"].replace('five', 5, inplace = True)
df["num-of-cylinders"].replace('six', 6, inplace = True)
df["num-of-cylinders"].replace('eight', 8, inplace = True)
df["num-of-cylinders"].replace('twelve', 12, inplace = True)

df["fuel-system"].replace('1bbl', 0, inplace = True)
df["fuel-system"].replace('2bbl', 1, inplace = True)
df["fuel-system"].replace('4bbl', 2, inplace = True)
df["fuel-system"].replace('idi', 3, inplace = True)
df["fuel-system"].replace('mfi', 4, inplace = True)
df["fuel-system"].replace('mpfi', 5, inplace = True)
df["fuel-system"].replace('spdi', 6, inplace = True)
df["fuel-system"].replace('spfi', 7, inplace = True)

df["horsepower"].replace('?',120,inplace=True)
df["peak-rpm"].replace('?',5000,inplace=True)
df["price"].replace('?',20000,inplace=True)
df["bore"].replace('?',3.00,inplace=True)
df["stroke"].replace('?',3.00,inplace=True)
df["fuel-system"].replace('spfi', 7, inplace = True)
df["normalized-losses"].replace('?',160,inplace=True)

df["fuel-type"].replace('gas',1,inplace=True)
df["fuel-type"].replace('diesel',2,inplace=True)


df["engine-location"].replace('front',1,inplace=True)
df["engine-location"].replace('rear',2,inplace=True)


print(df.dtypes)
plt.figure(figsize=(15,13))
sns.heatmap(df.corr(),annot=True)
plt.show()


##Exporting to csv
'''
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)
'''

##Generating descriptive stats
"""
#print(df.describe(include="all"))
#print(df.info)
"""

##Data Formatting, replacing ? to NAN
'''
df["price"].replace('?',np.nan, inplace = True)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)
'''

##Data Formatting, converting prices from Object to Int, dropping NaN values


df["price"].replace('?',np.nan, inplace = True)
df.dropna(subset=["price"], axis=0, inplace=True)
df["price"] = df["price"].astype("int")

##Data Formatting, converting peak-rpm from Object to Int, dropping NaN values

df["peak-rpm"].replace('?',np.nan, inplace = True)
df.dropna(subset=["peak-rpm"], axis=0, inplace=True)
df["peak-rpm"] = df["peak-rpm"].astype("int")

df["normalized-losses"].replace('?',np.nan, inplace = True)
df.dropna(subset=["normalized-losses"], axis=0, inplace=True)
df["normalized-losses"] = df["normalized-losses"].astype("int")

df["bore"].replace('?',np.nan, inplace = True)
df.dropna(subset=["bore"], axis=0, inplace=True)
df["bore"] = df["bore"].astype("float")

df["stroke"].replace('?',np.nan, inplace = True)
df.dropna(subset=["stroke"], axis=0, inplace=True)
df["stroke"] = df["stroke"].astype("float")

df["horsepower"].replace('?',np.nan, inplace = True)
df.dropna(subset=["horsepower"], axis=0, inplace=True)
df["horsepower"] = df["horsepower"].astype("int")

print(df.dtypes)
#print(df.info)
'''
'''
###Data Binning
'''
binwidth = int((max(df["price"])-min(df["price"]))/3)
bins = range(min(df["price"]), max(df["price"]), binwidth)
group_names = ['low','medium','high']
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)
df.dropna(subset=["price-binned"], axis=0, inplace=True)
'''

##Plotting Histogram from the binned value

'''
plt.hist(df["price"],bins=3)
plt.title("Price Bins")
plt.xlabel("Count")
plt.ylabel("Price")
plt.show()
'''

#TURNING CATEGORICAL VARIABLES INTO QUANTITATIVE VARIABLES
'''
df = (pd.get_dummies(df["fuel-type"]))
'''

#DESCRIPTIVE STATISTICS- Value_counts
'''
drive_wheels_counts = df["drive-wheels"].value_counts()
drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)
'''

#Box Plots


sns.boxplot(x="drive-wheels", y="symboling", data=df)
plt.show()
'''
#Scatterplot
y=df["engine-size"]
x=df["price"]
plt.scatter(x,y)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()
#Group by to visualize price based on drive-wheels and body style.
df_test = df[["drive-wheels", "body-style", "price"]]
df_group = df_test.groupby(['drive-wheels', 'body-style'], as_index = False).mean()
#Pivot Table to visualize price based on drive-wheels and body style.
df_pivot = df_group.pivot(index = 'drive-wheels', columns= 'body-style')
print(df_pivot)
#Heat Maps
plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
'''
#CORRELATION, Positive Linear Relationship between engine size and price
sns.regplot(x='drive-wheels', y='symboling', data=df)
plt.title("Scatterplot of drive-wheels vs Price")
plt.xlabel("drive-wheels")
plt.ylabel("symboling")
plt.ylim(-3,3)
plt.show()
#CORRELATION, Negetive Linear Relationship between highway-mpg and price
sns.regplot(x='highway-mpg', y='symboling', data=df)
plt.title("Scatterplot of highway-mpg vs symboling")
plt.xlabel("highway-mpg")
plt.ylabel("symboling")
plt.ylim(-3,3)
plt.show()
# WEAK CORRELATION between peak-rpm and price
sns.regplot(x='fuel-system', y='symboling', data=df)
plt.title("Scatterplot of fuel-system vs symboling")
plt.xlabel("fuel-system")
plt.ylabel("symboling")
plt.ylim(-3,3)
plt.show()

'''
# Simple Linear Model Estimator with Distribution plot


lm = LinearRegression()
X=df[["highway-mpg"]]
Y=df["symboling"]
lm.fit(X,Y)
Yhat1 = lm.predict(X)
b0 = lm.intercept_
b1 = lm.coef_
estimated = b0 + b1*X

ax1 = sns.distplot(df["symboling"],hist = False, color="r", label="Actual Value")
sns.distplot(Yhat1, hist = False, color="b", label="Fitted Value", ax=ax1)
plt.ylim(0,)
plt.show()
'''


# Multiple Linear Regression with Distribution plot
'''
lm = LinearRegression()
sv=svm.SVC(probability = True)
Z = df[["symboling", "make", "aspiration", "num-of-doors", "body-style", "drive-wheels",
         "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
         "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
         "highway-mpg", "price"]]
Y=df["symboling"]
model=sv.fit(Z,Y)
Yhat2 = sv.predict(Z)
print(Yhat2)
ax1 = sns.distplot(df["symboling"],hist = False, color="r", label="Actual Value")
sns.distplot(Yhat2, hist = False, color="b", label="Fitted Value", ax=ax1)
plt.ylim(0,)
plt.show()
'''
'''
#MLP
Z = df[["symboling", "make", "aspiration", "num-of-doors", "body-style", "drive-wheels",
         "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
         "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
         "highway-mpg", "price"]]
Y=df["symboling"]
mlp = MLPClassifier(solver='sgd',momentum=0.5,activation='relu',alpha=1e-4,hidden_layer_sizes=(100,80), random_state=1,max_iter=100000,verbose=10,learning_rate_init=0.0001,n_iter_no_change=100000)
mlp.fit(Z,Y)
print (mlp.score(Z,Y))
'''
#Decision Tree
x,y=df.iloc[:,2:],df.iloc[:,1]
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.3,random_state=0)
print(len(xtr))
print(len(xte))
dt=tree.DecisionTreeClassifier(criterion='entropy')
dt.fit(xtr,ytr)
qyte = dt.predict(xte)
train_error = np.mean(yte!=qyte)
print("train err is %f" % train_error)

'''
# Residual Plot

sns.residplot(df["highway-mpg"], df["symboling"])
plt.xlabel("highway-mpg")
plt.ylabel("symboling")
plt.show()
'''