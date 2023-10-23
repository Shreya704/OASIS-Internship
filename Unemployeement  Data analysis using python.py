#!/usr/bin/env python
# coding: utf-8

# In[117]:


#Importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ps
import plotly.express as px
import datetime as dt
import calendar
import warnings 
warnings.filterwarnings('ignore',category=DeprecationWarning)


# # DATA COLLECTION AND ANALYSIS

# In[118]:


data =pd.read_csv("Unemployment_Rate_upto_11_2020.csv")


# In[119]:


data.head(10)


# In[120]:


data.tail(10)


# In[121]:


data.info()


# In[122]:


data.columns


# In[123]:


data.rename(columns={'Region':'Places',' Date':'Date',' Estimated Unemployment Rate (%)':'Estimated Unemployment Rate (%)',' Estimated Employed':'Estimated Employed',' Estimated Labour Participation Rate (%)':'Estimated Labour Participation Rate (%)','Region.1':'Region'},inplace=True)


# In[124]:


data.head(10)


# In[125]:


data.isnull().sum()


# In[126]:


def summary(df):
    sum = pd.DataFrame( data.dtypes, columns=['dtypes'])
    sum['missing#'] =  data.isna().sum()
    sum['missing%'] = ( data.isna().sum().values*100)/len( data)
    sum['uniques'] =  data.nunique().values
    sum['count'] =  data.count().values
    desc = pd.DataFrame(df.describe().T)
    sum['min'] = desc['min']
    sum['max'] = desc['max']
    sum['mean'] = desc['mean']
    return sum

summary( data).style.background_gradient(cmap='YlOrRd')


# In[127]:


data.drop(columns=[' Frequency'],inplace=True)


# In[128]:


data.head(10)


# In[129]:


plt.figure(figsize=(10,6))
plt.xticks(rotation=35)
sns.barplot(data= data, x='Region', y='Estimated Employed',palette='Set2',order= data.groupby('Region')['Estimated Employed'].mean().sort_values().index)  
plt.title('Estimated Employed by Region') 
plt.tight_layout()


# In[130]:


plt.figure(figsize=(20,6))
plt.xticks(rotation=35)
sns.barplot(data= data, x='Places', y='Estimated Employed',palette='Set2',order= data.groupby('Places')['Estimated Employed'].mean().sort_values().index)  
plt.title('Estimated Employed by Places') 
plt.tight_layout()


# In[131]:


sns.set(style="whitegrid")
plt.figure(figsize=(18, 6))
sns.barplot(data=data, x='Places', y='Estimated Unemployment Rate (%)', order=df.groupby('Places')['Estimated Unemployment Rate (%)'].mean().sort_values().index)
plt.xlabel('Places')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('Estimated Unemployment Rate by Places (Sorted)')
plt.xticks(rotation=50)  
plt.tight_layout()
plt.show()


# In[132]:


unemplo_df = df[['Places', 'Region', 'Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']]
unemplo = unemplo_df.groupby(['Region', 'Places'])['Estimated Unemployment Rate (%)'].mean().reset_index()
fig = px.sunburst(unemplo, path=['Region', 'Places'], values='Estimated Unemployment Rate (%)',
                  color_continuous_scale='Plasma', title='Unemployment rate in each region and Place',
                  height=550, template='ggplot2')
fig.show()


# # 2019 to 2020

# In[133]:


data2['Date'] = pd.to_datetime(data2['Date'])


# In[134]:


data2 = pd.read_csv("unemploymentinIndia.csv")


# In[135]:


data2.shape


# In[136]:


data2.head(10)


# In[137]:


data2.rename(columns={'Region':'Places',' Date':'Date',' Estimated Unemployment Rate (%)':'Estimated Unemployment Rate (%)',' Estimated Employed':'Estimated Employed',' Estimated Labour Participation Rate (%)':'Estimated Labour Participation Rate'},inplace=True)


# In[138]:


data2['Date'] = pd.to_datetime(data2['Date'])



# In[139]:


data2['Month']=data2['Date'].dt.month_name()
data2['Year']=data2['Date'].dt.year


# In[140]:


data2.drop([' Frequency'],axis=1,inplace=True)


# In[141]:


data2.head()


# In[142]:


data2['Year'].value_counts()


# In[143]:


plt.figure(figsize=(22,7))
sns.barplot(data=data2,x='Places',y='Estimated Unemployment Rate (%)',order=data2.groupby('Places')['Estimated Unemployment Rate (%)'].mean().sort_values().index,hue='Year')
plt.title("Average Unemployment Rate In All Regions In Year(2019-2020)")
plt.xticks(rotation=65)  
plt.show()


# In[144]:


sns.barplot(data=data2,x='Area',y='Estimated Unemployment Rate (%)',hue='Year')
plt.show()


# # Mean Estimated Unemployment Rate Over Time

# In[145]:


grouped = data2.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
plt.figure(figsize=(17, 6))
plt.plot(grouped['Date'], grouped['Estimated Unemployment Rate (%)'], marker='o')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('Mean Estimated Unemployment Rate Over Time')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[ ]:




