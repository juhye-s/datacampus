#!/usr/bin/env python
# coding: utf-8

# ### 1. PCA
# #### 와인데이터의 전체 변수를 사용한 logistic regression 모델과 PCA로 차원을 축소한모델의 성능을 비교
# - the coef_ did not converge 메세지가 나오는 경우, 모델 만들때 max_iter=10000 추가 
#     - ex) LogisticRegression(solver="sag",max_iter=10000, multi_class="multinomial")

# In[2]:


import pandas as pd
import numpy as np

from sklearn import datasets


# In[4]:


wine= datasets.load_wine()
x=wine.data[:,[0,2]]
y=wine.target


# In[5]:


feature_name=[wine.feature_names[0],wine.feature_names[2]]
x_data=pd.DataFrame(x, columns=feature_name)
x_data.head()


# In[6]:


y_data=pd.DataFrame(y, columns=["target"])
y_data.head()


# In[7]:


from sklearn.decomposition import PCA


# In[8]:


pca=PCA(n_components=2)
pca.fit(x_data)


# In[9]:


PCscore=pca.transform(x_data)
PCscore[0:5]


# In[10]:


x_data[0:5]


# In[11]:


eigens_v=pca.components_.transpose()
print(eigens_v)


# In[12]:


mX=np.matrix(x)

for i in range(x.shape[1]):
    mX[:,i]=mX[:,i]-np.mean(x[:,i])
    
mX_df=pd.DataFrame(mX)


# In[13]:


(mX*eigens_v)[0:5]


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(PCscore[:,0],PCscore[:,1])
plt.show


# In[15]:


plt.scatter(mX_df[0],mX_df[1])
origin=[0],[0]
plt.quiver((0,0),(0,0), eigens_v[0,:],eigens_v[1,:], color=['r','b'], scale=3)
plt.show()


# In[16]:


plt.scatter(mX_df[0],mX_df[1],c=y)
origin=[0],[0]
plt.quiver((0,0),(0,0), eigens_v[0,:],eigens_v[1,:], color=['r','b'], scale=3)
plt.show()


# In[18]:


x2= wine.data
pca2=PCA(n_components=4)
pca2.fit(x2)
PCscore2=pca2.transform(x2)[:,0:2]


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[20]:


clf=LogisticRegression(solver="sag", multi_class="multinomial",max_iter=10000).fit(x2,y)


# In[26]:


clf2=LogisticRegression(solver="sag", multi_class="multinomial",max_iter=1000).fit(PCscore2,y)


# In[27]:


y_pred2=clf2.predict(PCscore2)


# In[28]:


y_pred=clf.predict(x2)


# In[29]:


confusion_matrix(y,y_pred2)


# ### 2. k-means
# #### 와인데이터를 군집화 하고 적절한 k를 확인

# In[49]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


# In[50]:


wine= datasets.load_wine()
wine


# In[51]:


x_data=wine.data[:,:2]
y_data=wine.target


# In[52]:


plt.scatter(x_data[:,0],x_data[:,1],c=y_data, cmap="spring")
plt.xlabel('alcohol')
plt.ylabel('malic_acid')


# In[53]:


from sklearn.cluster import KMeans


# In[54]:


km=KMeans(n_clusters=3, random_state=102)
km.fit(x_data)


# In[55]:


centers=km.cluster_centers_
new_labels=km.labels_


# In[65]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(x_data[:, 0], x_data[:, 1], c=new_labels, cmap='jet', edgecolor='k', s=150)

axes[0].set_xlabel('alcohol', fontsize=18)
axes[0].set_ylabel('malic_acid', fontsize=18)
axes[1].set_xlabel('alcohol', fontsize=18)
axes[1].set_ylabel('malic_acid', fontsize=18)

axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)


# In[66]:


wine_df=pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["target"]=pd.DataFrame(wine.target)
wine_df.head()


# In[67]:


wine_df["new_labels"]=new_labels
wine_df.head()


# In[68]:


wine_result=wine_df.groupby(["target","new_labels"])["alcohol"].count()
wine_result


# In[69]:


from sklearn.datasets import make_blobs


# In[72]:


X,y=make_blobs(n_samples=178, n_features=2,
               centers=3, random_state=10)


# In[73]:


plt.scatter(X[:,0],X[:,1],c='white', marker='o', edgecolor='black', s=50)
plt.show()


# In[74]:


y_km=km.labels_


# In[75]:


plt.scatter(X[y_km==0,0],X[y_km==0,1],c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='lightblue', marker='^', edgecolor='black', label='cluster 3')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250, marker='*',
            c='red', edgecolor='black', label='centroids')

plt.legend()
plt.grid()
plt.show()


# In[76]:


distortions=[]

for i in range(1,11):
    km=KMeans(n_clusters=i, random_state=102)
    km.fit(X)
    distortions.append(km.inertia_)
    
plt.plot(range(1,11),distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel("Distortion")
plt.show()
    


# ### 3. DBSCAN
# #### 와인데이터 군집화

# In[83]:


import pandas as pd
from sklearn import datasets
wine=datasets.load_wine()


# In[99]:


wine.feature_names


# In[105]:


wine.target_names


# In[109]:


#wine_df=pd.DataFrame(wine.data, columns=["alcohol","malic_acid","ash","alcalinity_of_ash"])
wine_df["labels"]=pd.DataFrame(wine.target)
wine_df.head()


# In[87]:


feature=wine_df[["alcohol","malic_acid","ash","alcalinity_of_ash"]]
feature.head()


# In[88]:


from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


# In[89]:


model=DBSCAN(eps=0.5, min_samples=5)
clst=model.fit_predict(feature)


# In[90]:


predict=pd.DataFrame(clst)
predict.columns=['predict']


# In[91]:


wine_pred=feature.join(predict)
wine_pred


# In[92]:


sns.pairplot(wine_pred, hue='predict')
plt.show()


# In[107]:


sns.pairplot(wine_df,hue='labels')
plt.show


# In[94]:


wine_df["predict"]=predict
wine_df.head()


# In[95]:


wine_result=wine_df.groupby(["labels", "predict"])["sepal length"].count()
wine_result


# In[ ]:




