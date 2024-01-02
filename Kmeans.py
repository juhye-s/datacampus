#!/usr/bin/env python
# coding: utf-8

# ## K-means
# #### iris 분류 

# In[2]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


iris=datasets.load_iris()


# In[4]:


iris


# In[5]:


x_data=iris.data[:,:2]
y_data=iris.target


# In[9]:


plt.scatter(x_data[:,0],x_data[:,1],c=y_data, cmap="spring")
plt.xlabel('Speal Length')
plt.ylabel('Sepal Width')


# In[8]:


from sklearn.cluster import KMeans


# In[10]:


km=KMeans(n_clusters=3, random_state=102)
km.fit(x_data)


# In[11]:


centers=km.cluster_centers_
print(centers)


# In[12]:


new_labels=km.labels_


# In[13]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(x_data[:, 0], x_data[:, 1], c=new_labels, cmap='jet', edgecolor='k', s=150)

axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)

axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)


# In[14]:


iris_df=pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"]=pd.DataFrame(iris.target)
iris_df.head()


# In[15]:


iris_df["new_labels"]=new_labels
iris_df.head()


# In[16]:


iris_result=iris_df.groupby(["target","new_labels"])["sepal length (cm)"].count()
iris_result


# In[ ]:





# ####  make_plobs() : 군집화용 데이터 생성

# In[17]:


from sklearn.datasets import make_blobs


# In[19]:


X,y=make_blobs(n_samples=150, n_features=2,
               centers=3, random_state=10)


# In[23]:


plt.scatter(X[:,0],X[:,1],c='white', marker='o', edgecolor='black', s=50)
plt.show()


# In[24]:


km=KMeans(n_clusters=3, random_state=102)
km.fit(X)


# In[25]:


y_km=km.labels_


# In[26]:


y_km


# In[29]:


plt.scatter(X[y_km==0,0],X[y_km==0,1],c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='lightblue', marker='^', edgecolor='black', label='cluster 3')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250, marker='*',
            c='red', edgecolor='black', label='centroids')

plt.legend()
plt.grid()
plt.show()


# - #### k 를 4로 할경우 

# In[30]:


km2=KMeans(n_clusters=4, random_state=102)
y_km2=km2.fit_predict(X)


# In[31]:


plt.scatter(X[y_km2==0,0],X[y_km2==0,1],c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(X[y_km2==1,0],X[y_km2==1,1],c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(X[y_km2==2,0],X[y_km2==2,1],c='lightblue', marker='^', edgecolor='black', label='cluster 3')
plt.scatter(X[y_km2==3,0],X[y_km2==3,1],c='gray', marker='d', edgecolor='black', label='cluster 4')

plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],s=250, marker='*',
            c='red', edgecolor='black', label='centroids')

plt.legend()
plt.grid()
plt.show()


# In[32]:


distortions=[]

for i in range(1,11):
    km=KMeans(n_clusters=i, random_state=102)
    km.fit(X)
    distortions.append(km.inertia_)
    
plt.plot(range(1,11),distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel("Distortion")
plt.show()
    


# In[ ]:




