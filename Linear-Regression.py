#!/usr/bin/env python
# coding: utf-8

# In[111]:


#importing necessary packages behorehand

#for mathematical and matrix operations
import numpy as np
import pandas as pd

#for data visualisation
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

#for pre-processing data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import random


# In[112]:


# importing the training dataset
train_df = pd.read_csv("train.csv")

#description of the dataset in terms of basic measures
train_df.describe()


# In[113]:


#finding out the data types of all fields
train_df.dtypes


# In[114]:


#importing the training dataset
test_df = pd.read_csv("test.csv")

#description of the dataset in terms of basic measures
test_df.describe()


# In[115]:


df = train_df
df.describe()


# In[116]:


#to check if any values of any of the fields are null.

df.isnull().sum()


# In[117]:


df.drop(['post_day', 'basetime_day' ], axis = 1)

#make dummy variables for them later and add these fields too.


# In[118]:


#making a boxplot for outlier detection
matplotlib.pyplot.boxplot(df.target)


# In[125]:


#finding out the correlation between features
corr1 = df.corr()


# In[127]:


corr1


# In[128]:


#plotting a heat map for easier detection of correlation amongst variables
plt.subplots(figsize=(10,10))
sns.heatmap(df.corr())


# In[129]:


### selecting top 20 attributes with respect to correlation
feature_list = corr1["target"].sort_values().tail(30).head(20) 
feature_list.index


# In[130]:


df[feature_list.index].corr()


# In[131]:


#number of instances for different count of comments
print(df['target'].value_counts())


# In[132]:


#Bar chart of independent variables with respect to output
labels = df['target'].astype('category').cat.categories.tolist()
counts = df['target'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[133]:



X = df.drop(['post_day', 'basetime_day' ], axis = 1)
# deciding on the final features
Features_final = ['base_time', 'page_category', 'post_length', 'h_target', 'promotion',
       'page_checkin', 'page_likes', 'c3', 'c5', 'share_count',
       'daily_crowd', 'F1', 'c1','target']

df_final = df[Features_final]
df_final.describe()


# In[134]:


# outlier removal by gaussian score
df_final = df_final[(np.abs(stats.zscore(df_final)) < 3).all(axis=1)]

df_final.describe()


# In[135]:


df_final.head()


# In[136]:


X = df_final.iloc[:,0:13]
Y = df_final["target"]
## Splitting data train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.1)

#applying feature scaling, specifically min-max scaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#using training data's min and max for scaling the testing data
X_train.shape


# In[151]:


#simple gradient descent algorithm

def gradient_descent(alpha, x, y, numIterations):
    # number of samples
    m = x.shape[0] 
    theta = np.ones(14)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
         ## Cost Function J to be calculated
        J = np.sum(loss ** 2) / (2 * m) 
        print("iter %s | J: %.3f" % (iter, J)) 
        gradient = np.dot(x_transpose, loss) / m 
        ## Batch update
        theta = theta - alpha * gradient 
        print(theta)
    return theta

if __name__ == '__main__':

    x, y = X_train, Y_train 
    m, n = np.shape(x)
    x = np.c_[ np.ones(m), x] # insert column
    alpha = 0.30  # learning rate
    theta = gradient_descent(alpha, x, y, 3000)


# In[152]:


def gradient_descent(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    x = np.c_[ np.ones(m), x]
    theta = np.ones(14)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        ## Cost Function J to be calculated
        J = np.sum(loss ** 2) / (2 * m)  
        gradient = np.dot(x_transpose, loss) / m   
        theta = theta - alpha * gradient 
    theta1 = theta
    y_train_pred = np.dot(x,theta1)
    rms_train = ((((y_train_pred - np.array(y))**2).sum())/m)**0.5 
    x1 = np.c_[ np.ones(X_test.shape[0]), X_test]
    y_test_pred = np.dot(x1,theta)
    rms_test = ((((y_test_pred - np.array(Y_test))**2).sum())/(x1.shape[0]))**0.5 
    xyz = (alpha,rms_train,rms_test)
    return xyz


# In[153]:


#running the algorithm for different values of alpha

df_exp1 = pd.DataFrame()
for alpha in (0.01,0.1,0.3,1.1,1.15,1.2):
    alpha,rms_train,rms_test=gradient_descent(alpha, X_train, Y_train, 3000)
    dict={"alpha" : [alpha],"rms_train" :[rms_train] ,"rms_test":[rms_test]}
    df1=pd.DataFrame(data = dict)
    df_exp1 = df_exp1.append(df1,ignore_index = True)


# In[154]:


df_exp1


# In[155]:


plt.plot(df_exp1.alpha,df_exp1.rms_train)
plt.plot(df_exp1.alpha,df_exp1.rms_test)
plt.axis([0, 1.5, 7.5, 9.5])
plt.xlabel("alpha-->")
plt.ylabel("root mean square error")
plt.legend()
plt.show()


# In[156]:


#saving the value of cost function at each step
costs = list()
#regularised gradient descent with lambda user-provided
def gradient_descent(alpha, x, y, numIterations, lmbda):
    m = x.shape[0] # number of samples
    x = np.c_[ np.ones(m), x]
    theta = np.ones(14)
    x_transpose = x.transpose()
    
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = (np.sum(loss ** 2) / (2 * m)) + ((lmbda * np.sum(theta[1:]**2) )/ (2 * m)) 
        costs.append(J)
        gradient = np.dot(x_transpose, loss) / m   
        theta = theta - alpha * gradient 
        
    theta1 = theta
    y_train_pred = np.dot(x,theta1)
    # root mean square error
    rms_train = ((((y_train_pred - np.array(y))**2).sum())/m)**0.5
    #mean square error
    mse_train= ((((y_train_pred - np.array(y))**2).sum())/m)
    
    x1 = np.c_[ np.ones(X_test.shape[0]), X_test]
    y_test_pred = np.dot(x1,theta)
    
    
    mse_test = ((((y_test_pred - np.array(Y_test))**2).sum())/(x1.shape[0]))
    rms_test = ((((y_test_pred - np.array(Y_test))**2).sum())/(x1.shape[0]))**0.5 ## root mean square error
    xyz = (alpha,rms_train,rms_test,mse_train, mse_test)
    return xyz


# In[157]:


df_exp1 = pd.DataFrame()
for alpha in (0.005, 0.05, 0.025,0.01,0.1,0.3):
    alpha,rms_train,rms_test, mse_train, mse_test=gradient_descent(alpha, X_train, Y_train, 10000,0.5)
    dict={"alpha" : [alpha],"rms_train" :[rms_train] ,"rms_test":[rms_test], "mse_train": [mse_train],"mse_test": [mse_test]}
    df1=pd.DataFrame(data = dict)
    df_exp1 = df_exp1.append(df1,ignore_index = True)


# In[158]:


df_exp1


# In[159]:


costs


# In[160]:


plt.plot(df_exp1.alpha,df_exp1.rms_train)
plt.plot(df_exp1.alpha,df_exp1.rms_test)
plt.axis([0.005, 1.5, 7.5, 9.5])
plt.xlabel("alpha-->")
plt.ylabel("root mean square error")
plt.legend()
plt.show()


# In[ ]:




