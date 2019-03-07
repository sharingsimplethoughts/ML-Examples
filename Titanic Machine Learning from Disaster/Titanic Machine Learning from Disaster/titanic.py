
# coding: utf-8

# In[109]:


pwd


# In[110]:


import pandas as pd
import numpy as np


# In[95]:


import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('train.csv')


# In[4]:


dataset


# In[111]:


dataset.shape


# In[6]:


dataset.isnull()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset = dataset.fillna(value={'Embarked': ' '})


# # Spliting Dependent and Independent Variable

# In[9]:


x = dataset.iloc[:,[0,2,4,5,6,7,9,11]].values


# In[10]:


y = dataset.iloc[:,1].values


# In[11]:


y


# In[12]:


x


# # Handlig NaN value

# In[13]:


from sklearn.preprocessing import Imputer


# In[14]:


imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)


# In[15]:


imputer = imputer.fit(x[:,3:4])


# In[16]:


x[:,3:4] = imputer.transform(x[:,3:4])


# # Encoding Categorical data and String data with Dummy 

# In[17]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[18]:


labelencoder = LabelEncoder()


# In[19]:


x[:,2] = labelencoder.fit_transform(x[:,2])


# In[20]:


x[:,7] = labelencoder.fit_transform(x[:,7])


# In[21]:


x


# In[22]:


y


# In[23]:


onehotencoder = OneHotEncoder(categorical_features = [2,7])


# In[24]:


x = onehotencoder.fit_transform(x).toarray()


# # Splitting the dataset to train and test set
# it is not required if data is very less, like 10 rows. we can directly use x as training set. 
# it may be also not required for SVR regression as svr is used to predict one value against one value

# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(x,y)


# # Feature Scaling
# in below code we dont need to fit x_test as sc already fitted to x_train

# In[27]:


from sklearn.preprocessing import StandardScaler


# In[28]:


sc = StandardScaler()


# In[29]:


x_train = sc.fit_transform(x_train)


# In[30]:


x_test = sc.transform(x_test)


# In[64]:


x


# In[65]:


x_test


# In[66]:


x_train


# In[67]:


y


# In[68]:


y_train


# In[69]:


y_test


# # Linear Regression
# Automatic feature scalling.
# 
# #### It is not possible to apply here as we have many independent columns and we are predicting only 0 or 1

# In[90]:


from sklearn.linear_model import LinearRegression


# In[91]:


regressor = LinearRegression()


# In[92]:


regressor.fit(x_train, y_train)


# In[93]:


y_pred = regressor.predict(x_test)


# In[177]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_test, color = 'blue')
plt.title('Titanic')
plt.xlabel('many columns of x')
plt.ylabel('y')
plt.show()


# # Multiple Linear Regression
# Automatic feature scalling.
# for every independent variable we need a separate dimention. here we have 12 columns. so it is hard to visually represent 12 dimentions. here I have tried to create a plot with 1 independent variable only.
# X and Y must be of same size.
# 
# It is not possible to apply Multiple Linear Regression here as we are predicting only 0 or 1. so the graph will be meaningless

# In[75]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression


# In[76]:


regressor = LinearRegression()


# In[77]:


regressor.fit(x_train, y_train)


# In[78]:


# Predicting the Test set results
y_pred = regressor.predict(x_test) 


# In[180]:


plt.scatter(x_test[:,11], y_test, color = 'red')
plt.plot(x_test[:,11], y_test, color = 'blue')
plt.title('testing')
plt.xlabel('x - index 11')
plt.ylabel('y')
plt.show()


# In[79]:


y_pred #no where near to y_test. y_test is totaly 0 and 1


# # Polynomial Regression 
# Automatic feature scalling
# X and Y must be of same size

# In[80]:


from sklearn.preprocessing import PolynomialFeatures


# In[81]:


poly_reg = PolynomialFeatures()


# In[82]:


x_poly = poly_reg.fit_transform(x_train)        


# In[89]:


x_train


# In[85]:


poly_reg.fit(x_poly, y_train)


# In[ ]:


lin_reg_2.predict(poly_reg.fit_transform(6.5))


# # SVR Regression
# ->No Automatic feature scalling.
# ->need to feature scale y or y_train also. It will give predicted y_test in encoded format. so at last we need to inverse the encoding again for the y_test. 
# ->X and Y must be of same size.

# In[41]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()


# In[42]:


x_train = sc_x.fit_transform(x_train)


# In[56]:


x_test = sc_x.fit_transform(x_test)


# In[44]:


y_train = np.reshape(y_train, (len(y_train),-1))


# In[45]:


y_train = sc_y.fit_transform(y_train)


# In[49]:


y_train = np.asarray(y_train).reshape(-1)


# In[50]:


from sklearn.svm import SVR


# In[51]:


regressor = SVR(kernel = 'rbf')


# In[52]:


regressor.fit(x_train, y_train)


# In[57]:


y_pred = regressor.predict(x_test)


# In[59]:


y_pred = sc_y.inverse_transform(y_pred)


# In[60]:


y_pred


# In[55]:


y_test

