#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data_train=pd.read_csv("credit_train.csv" , encoding='utf-8')
data_train.head()


# In[ ]:


data_train.info()


# In[ ]:


data_train.describe()


# In[ ]:


data_train.describe().T


# In[ ]:


data_train.columns


# In[18]:


data_train.shape


# In[20]:


data_train.head()


# In[21]:


data_train.info()


# In[ ]:


#A function to calculate and print out the missing value and it percentage 


# In[22]:


def calculate_null_values(dataframe):
    d_frame = dataframe
    # get the sum of the null value of  each column 
    d_frame_null_values = pd.DataFrame(dataframe.isna().sum())
    # reset the dataframe index
    d_frame_null_values.reset_index(inplace=True)
    # add colume header to the dataframe
    d_frame_null_values.columns = ['Field_names', 'Null_value']
    #calculate the percentage of null or missing values 
    d_frame_null_value_percentage = dataframe.isnull().sum() / len(dataframe) * 100
    d_frame_null_value_percentage = pd.DataFrame(d_frame_null_value_percentage)
    d_frame_null_value_percentage.reset_index(inplace=True)
    d_frame_null_value_percentage = pd.DataFrame(d_frame_null_value_percentage)
    d_frame_null_values['Null_values_percentage'] = d_frame_null_value_percentage[0]
    return d_frame_null_values


# In[23]:


calculate_null_values(data_train)


# In[27]:


plt.figure(figsize=(15,8))
sns.heatmap(data_train.isnull())


# In[28]:


#Data preprocessing
data_train_1 = data_train.drop(labels=['Loan ID', 'Customer ID'], axis=1)


# In[30]:


data_train.shape


# In[29]:


data_train_1.shape


# In[32]:


#Dealing with missing values
# about 50 % of it is missing 
data_train_1.drop(columns = 'Months since last delinquent', axis=1, inplace=True)
calculate_null_values(data_train_1)


# In[33]:


data_train_1[data_train_1['Years of Credit History'].isnull() == True]


# In[35]:


## We note that the last 514 values are misiing values 

data_train_1.drop(data_train.tail(514).index, inplace=True)
# drop last 514 rows
calculate_null_values(data_train_1)


# In[38]:


for i in data_train_1['Maximum Open Credit'][data_train_1['Maximum Open Credit'].isnull() == True].index:
    data_train_1.drop(labels=i, inplace=True)


# In[39]:


for i in data_train_1['Tax Liens'][data_train_1['Tax Liens'].isnull() == True].index:
    data_train_1.drop(labels=i, inplace=True)


# In[40]:


for i in data_train_1['Bankruptcies'][data_train_1['Bankruptcies'].isnull() == True].index:
    data_train_1.drop(labels=i, inplace=True)


# In[41]:


calculate_null_values(data_train_1)


# In[42]:


data_train.shape


# In[43]:


data_train_1.shape


# In[44]:


sns.displot(data = data_train_1 , x = "Credit Score" ,  kind="kde",)


# In[49]:


fill_list = data_train_1['Credit Score'].dropna()
data_train_1['Credit Score'] = data_train['Credit Score'].fillna(pd.Series(np.random.choice(fill_list , size = len(data_train_1.index))))


# In[50]:


data_train_1.dropna(axis = 0, subset = ['Credit Score'], inplace = True)


# In[51]:


sns.displot(data = data_train_1 , x = "Credit Score" ,  kind="kde",)


# In[53]:


sns.kdeplot(data_train_1['Annual Income'],
                color="Red", shade = True)


# In[54]:


fill_list = data_train_1['Annual Income'].dropna()
data_train_1['Annual Income'] = data_train_1['Annual Income'].fillna(pd.Series(np.random.choice(fill_list , size = len(data_train_1.index))))


# In[55]:


data_train_1.dropna(axis = 0, subset = ['Annual Income'], inplace = True)


# In[56]:


sns.kdeplot(data_train_1['Annual Income'],
                color="Red", shade = True)


# In[57]:


plt.figure(figsize=(16,8))

sns.countplot(data_train_1['Years in current job'])


# In[58]:


data_train_1['Years in current job'].fillna('10+ years', inplace=True)
# fill with '10+ years'.


# In[59]:


calculate_null_values(data_train_1)


# In[60]:


data_train_1.shape


# In[61]:


plt.figure(figsize=(16,10))
sns.heatmap(data_train.isnull())


# In[62]:


plt.figure(figsize=(16,10))
sns.heatmap(data_train_1.isnull())


# In[63]:


# Now there is no mising value 

###Drop the duplicated value and edit some values


# In[64]:


data_train_1.duplicated().sum()


# In[65]:


data_train_1.drop_duplicates(inplace=True)


# In[66]:


data_train_1.shape


# In[67]:


#Function to view the value counts for the whole data
def v_counts(dataframe):
    for i in dataframe :
        print(dataframe[i].value_counts())
        print("_____________________________________________________________________________")


# In[68]:


v_counts(data_train_1)


# In[69]:


data_train_1.Purpose = data_train_1.Purpose.str.replace('other','Other')


# In[70]:


from sklearn.preprocessing import LabelEncoder


# In[71]:


l_encoder = LabelEncoder()


# In[72]:


data_train_1['Loan Status'] = l_encoder.fit_transform(data_train_1['Loan Status'])
data_train_1['Term'] = l_encoder.fit_transform(data_train_1['Term'])
data_train_1['Years in current job'] = l_encoder.fit_transform(data_train_1['Years in current job'])
data_train_1['Home Ownership'] = l_encoder.fit_transform(data_train_1['Home Ownership'])
data_train_1['Purpose'] = l_encoder.fit_transform(data_train_1['Purpose'])


# In[73]:


v_counts(data_train_1)


# In[74]:


#Split the data


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


x= data_train_1.drop(['Loan Status' ] , axis=1).values
y = data_train_1['Loan Status'].values


# In[101]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.25 , random_state= 42)


# In[102]:


print(x_train.shape , x_test.shape)


# In[103]:


#Data Scalling
from sklearn.preprocessing import StandardScaler 


# In[104]:


scalar = StandardScaler()


# In[105]:


x_train = scalar.fit_transform(x_train)


# In[106]:


x_test = scalar.fit_transform(x_test)


# In[107]:


x_train.shape


# In[108]:


#Logistic Model


# In[109]:


from sklearn.linear_model import LogisticRegression


# In[110]:


lg = LogisticRegression(max_iter = 500)


# In[96]:


#lg = LogisticRegression(max_iter=1500)


# In[111]:


lg.fit(x_train , y_train)


# In[ ]:





# In[112]:


lg.score(x_train , y_train)


# In[113]:


lg.score(x_test , y_test)


# In[114]:


lg.intercept_


# In[115]:


lg.coef_


# In[116]:


y_predict = lg.predict(x_test)
df = pd.DataFrame({"Y_test": y_test , "Y_predict" : y_predict})
df


# In[117]:


plt.figure(figsize=(15,8))
plt.plot(df[:100])


# In[ ]:


###Test the model


# In[118]:


pass1 = [0,20,1,0,1,254,3,15,4,2,1,4 ,2,4,5]


# In[119]:


pass2 = [1,40,1,1,0,300,2,10,2,4,0,2,1,3,1]


# In[120]:


lg.predict([pass1])


# In[121]:


lg.predict([pass2])


# In[ ]:


#KNN model

from sklearn.neighbors import KNeighborsClassifier


# In[128]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[129]:


knn.fit(x_train , y_train)


# In[ ]:


knn.score(x_train , y_train )


# In[ ]:


knn.score(x_test , y_test )


# In[ ]:


y_predict = knn.predict(x_test)
df = pd.DataFrame({"Y_test": y_test , "Y_predict" : y_predict})
df.head(10)


# In[126]:


#Test the model
knn.predict([pass1])


# In[127]:


knn.predict([pass2])


# In[ ]:


#Desicion Tree model


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier( max_depth= 10 , max_features= 15)


# In[ ]:


dt.fit(x_train , y_train)


# In[ ]:


dt.score(x_train , y_train)


# In[ ]:


dt.score(x_test , y_test)


# In[ ]:


y_predict = dt.predict(x_test)
df = pd.DataFrame({"Y_test": y_test , "Y_predict" : y_predict})
df.head(10)


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(df[:100])


# In[ ]:


feature_importances = rf.feature_importances_
# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = data_train.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Create a bar plot of the feature importances
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(sorted_importances, sorted_feature_names)


# In[ ]:


### NB


# In[133]:


from sklearn.naive_bayes import GaussianNB


# In[134]:


gnb = GaussianNB()


# In[135]:


gnb.fit(x_train,y_train)


# In[136]:


gnb.score(x_train,y_train)


# In[137]:


gnb.score(x_test,y_test)


# In[138]:


gnb.predict(x_test)
gnb = pd.DataFrame({"Y_test": y_test , "Y_predict" : y_predict})
gnb.head(20)


# In[ ]:


pass1 = [0,20,1,0,1,254,3,15,4,2,1,4 ,2,4,5]


# In[ ]:


pass2 = [1,40,1,1,0,300,2,10,2,4,0,2,1,3,1]


# In[ ]:


gnb.predict([pass1])


# In[ ]:


gnb.predict([pass2])


# In[ ]:


## SVC
from sklearn.svm import SVC


# In[ ]:


svc = SVC( C=2, kernel='rbf'  , probability=True)
#{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}


# In[ ]:


svc.fit(x_train,y_train)


# In[ ]:


svc.score(x_train,y_train)


# In[ ]:


svc.score(x_test,y_test)


# In[ ]:


#svc.predict(x_test)
svc.predict(x_test)
DF = pd.DataFrame({"Y_test": y_test , "Y_predict" : y_predict})
DF.head(10)


# In[ ]:


svc.predict_proba(x_test)
PF = pd.DataFrame({"Y_test": y_test , "Y_predict" : y_predict})
PF.head(5)


# In[ ]:


from collections import Counter
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids


# In[ ]:


from imblearn.under_sampling import ClusterCentroids
undersampler = ClusterCentroids()


# In[ ]:


X_smote, y_smote = undersampler.fit_resample(X_train, y_train)


# In[ ]:


y_smote.value_counts()


# In[ ]:


x_smote


# In[ ]:




