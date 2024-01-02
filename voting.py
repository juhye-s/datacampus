#!/usr/bin/env python
# coding: utf-8

# ### Voting
# #### 위스콘신 유방암 데이터

# - from sklearn.ensemble import VotingClassifier

# In[2]:


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[3]:


cancer=datasets.load_breast_cancer()


# In[4]:


print(cancer.DESCR)


# In[7]:


x_df=pd.DataFrame(cancer.data, columns=cancer.feature_names)
x_df.head()


# In[32]:


y_df=pd.DataFrame(cancer.target, columns=["target"])
y_df.head()


# In[33]:


x_df.info()


# In[34]:


y_df["target"].value_counts()


# In[35]:


x_train, x_test, y_train, y_test=train_test_split(cancer.data, cancer.target,
                                                 train_size=0.8, test_size=0.2,
                                                 random_state=156)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


# In[37]:


lr_clf=LogisticRegression(max_iter=1000)
knn_clf=KNeighborsClassifier(n_neighbors=5)


# In[38]:


hvot_clf=VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)], voting='hard')
hvot_clf.fit(x_train, y_train)


# In[40]:


svot_clf=VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)], voting='soft')
svot_clf.fit(x_train,y_train)


# In[41]:


pred_hvot=hvot_clf.predict(x_test)
pred_svot=svot_clf.predict(x_test)


# In[42]:


print(accuracy_score(y_test, pred_hvot))
print(accuracy_score(y_test, pred_svot))


# In[43]:


lr_clf.fit(x_train, y_train)
pred_lr=lr_clf.predict(x_test)
acc_lr=accuracy_score(y_test, pred_lr)
print(acc_lr)


# In[46]:


knn_clf.fit(x_train, y_train)
pred_knn=knn_clf.predict(x_test)
acc_knn=accuracy_score(y_test, pred_knn)
print(acc_knn)

