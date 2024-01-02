#!/usr/bin/env python
# coding: utf-8

# 자연어처리 과제1

# In[1]:


import re


# In[52]:


text="제 휴대폰 번호는 010-1234-5678이고, 전화 번호는 02.987.6543입니다."
re.findall("\d+",text)


# In[35]:


webs=("""http://www.test.co.kr,
https://www.test1.com.
http://www.test.com,
ftp://www.test.com,
http://www.google.com,
https://www.homepage.com""")
re.findall("http.+",webs)

