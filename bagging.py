#!/usr/bin/env python
# coding: utf-8

# ### Bagging
# #### 위스콘신 유방암 데이터

# - from sklearn.ensemble import BaggingClassifier

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split


# In[2]:


cancer=datasets.load_breast_cancer()


# In[3]:


print(cancer.DESCR)


# In[4]:


x_train, x_test, y_train, y_test=train_test_split(cancer.data, cancer.target,
                                                 train_size=0.8, test_size=0.2,
                                                 random_state=156)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error


# In[6]:


lr_clf=LogisticRegression(max_iter=10000)
lr_clf.fit(x_train, y_train)

pred_lr=lr_clf.predict(x_test)


# In[8]:


from sklearn.ensemble import BaggingClassifier

bag_clf=BaggingClassifier(base_estimator=lr_clf,
                         n_estimators=5,
                         verbose=1)


# In[9]:


lr_clf_bag=bag_clf.fit(x_train,y_train)
pred_lr_bag=lr_clf_bag.predict(x_test)


# In[10]:


pred_lr_bag


# In[11]:


print(accuracy_score(y_test, pred_lr_bag))
print(mean_squared_error(y_test, pred_lr_bag))


# In[12]:


from sklearn.tree import DecisionTreeClassifier

dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)
pred_dt=dt_clf.predict(x_test)

print(accuracy_score(y_test, pred_dt))
print(mean_squared_error(y_test, pred_dt))


# In[18]:


bag_dt_clf=BaggingClassifier(base_estimator=dt_clf,
                            n_estimators=100,
                             verbose=1)


# In[17]:


bag_dt_clf.fit(x_train, y_train)
pred_dt_bag=bag_dt_clf.predict(x_test)

print(accuracy_score(y_test, pred_dt_bag))
print(mean_squared_error(y_test, pred_dt_bag))


# ### RandomForest

# - from sklearn.ensemble import RandomForestClassifier
# - from sklearn.model_selection import GridSearchCV

# In[20]:


from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier(n_estimators=5, max_depth=3, random_state=103, verbose=1)

rf_clf.fit(x_train, y_train)
pred=rf_clf.predict(x_test)
print(accuracy_score(y_test, pred))


# In[23]:


rf_clf2=RandomForestClassifier(n_estimators=500,
                              max_depth=3, random_state=103,
                              verbose=1)
rf_clf2.fit(x_train, y_train)
pred2=rf_clf2.predict(x_test)
print(accuracy_score(y_test, pred2))


# In[24]:


rf_clf3=RandomForestClassifier(n_estimators=500,
                              max_depth=10, random_state=103,
                              verbose=1)
rf_clf3.fit(x_train, y_train)
pred3=rf_clf3.predict(x_test)
print(accuracy_score(y_test, pred3))


# In[27]:


from sklearn.model_selection import GridSearchCV

rf_clf4=RandomForestClassifier()
rf_clf4


# In[32]:


params={'n_estimators':[10,100,500,1000],
       'max_depth':[3,5,10,15]}
rf_clf4=RandomForestClassifier(random_state=103,
                               n_jobs=-1,
                               verbose=1)

grid_cv=GridSearchCV(rf_clf4,
                    param_grid=params,
                    n_jobs=-1,
                    verbose=1)

grid_cv.fit(x_train, y_train)

print('최적 하이퍼 파라미터:',grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))


# In[ ]:


rf_clf5=RandomForestClassifier(n_estimators=1000,
                              max_depth=10, random_state=103,
                              verbose=1)
rf_clf5.fit(x_train, y_train)
pred5=rf_clf5.predict(x_test)
print(accuracy_score(y_test, pred5))


# In[ ]:




