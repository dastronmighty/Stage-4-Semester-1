#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

pathlib.Path('./output').mkdir(exist_ok=True)


# In[16]:


print('*' * 20)
print('Question 1')
print('*' * 20)
sdata = pd.read_csv("./specs/SensorData_question1.csv")

print('Creating "Original Input3" and "Original Input12"')
sdata["Original Input3"] = sdata["Input3"]
sdata["Original Input12"] = sdata["Input12"]
print('='*40)
print("")
print('Running Z-Score Normalization on Input 3')
sdata["Input3"] = ((sdata["Input3"] - sdata["Input3"].mean()) / sdata["Input3"].std())

print('='*40)
print("")

print('Running Min Max Normalization on Input12')
mmscaler = MinMaxScaler()
sdata["Input12"] = mmscaler.fit_transform(sdata["Input12"].values.reshape(-1,1))

print('='*40)
print("")

print('Averaging inputs by row and storing in "Average Input"')
sdata['Average Input'] = sdata.iloc[:,0:12].mean(axis=1)

print('='*40)
print("")

print("Saving new data in '/output/question1_out.csv'")
sdata.to_csv('./output/question1_out.csv', index=False, float_format='%g')

print('*' * 20)
print('Done!')
print('*' * 20)


# In[17]:


print('*' * 20)
print('Question 2')
print('*' * 20)
print("")
DNAData = pd.read_csv("./specs/DNAData_question2.csv")

print('Q2 - Part 1:')
print('\tExtracting Data...')
x = DNAData.values
pca_dna = PCA(n_components=0.95, svd_solver='full')
pca_dna.fit(x)
principalComponents_DNAdata = pca_dna.transform(x)
print('\tFit & Transforming...')
ncomp = pca_dna.n_components_
explvar = round(pca_dna.explained_variance_ratio_.sum(),3)
print('\tDone!')
print(f"\tComponents = {ncomp}\n\tTotal explained variance = {explvar}%")

# So what does this actually say?
# Well since we end up with 22 components this means we can transform the data using these 22 
# Principal Components and still end up with data which is about 95% similar to the original 
# keep in mind when we transform we are doing a linear combination over all the data into 22 new columns instead of 59
# NOTE: Still a n22 Space so we do't actually know whats what anymore really 


print('='*40)
print("")
print('Q2 - Part 2:')
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
print('\tDiscretizing using Equal Width')
dnaDwidth = discretizer.fit_transform(principalComponents_DNAdata)
print('\tAdding "width columns"')
wframe = pd.DataFrame(dnaDwidth)
for i in range(dnaDwidth.shape[1]):
    DNAData["pca"+str(i)+"_width"] = wframe[i]
    
print('='*40)
print("")
print('Q2 - Part 3:')
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
print('\tDiscretizing using Equal Width')
dnaDfreq = discretizer.fit_transform(principalComponents_DNAdata)
print('\tAdding "width columns"')
dframe = pd.DataFrame(dnaDfreq)
for i in range(dnaDwidth.shape[1]):
    DNAData["pca"+str(i)+"_freq"] = dframe[i]
print('='*40)
print("")
    
print("Saving new data in '/output/question2_out.csv'")
DNAData.to_csv('./output/question2_out.csv', index=False, float_format='%g')
    
print("")
print('*' * 20)
print('Done!')
print('*' * 20)


# In[ ]:





# In[ ]:




