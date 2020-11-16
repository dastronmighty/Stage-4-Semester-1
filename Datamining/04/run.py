#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""
AUTHOR: EOGHAN HOGAN
        EOGHAN.HOGAN@UCDCONNECT.IE
        17335293
"""

#STANDARD IMPORTS
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn import preprocessing

from matplotlib import pyplot as plt

import pathlib

# SETUP OUTPUT PATH
pathlib.Path('./output').mkdir(exist_ok=True)


# In[22]:


print("="*40)
print(f"Question 1")
print("="*40, "\n")

print("*"*5, "Question 1 - 1\n", "*"*5)
# read the csv and sort the values based on midterm
marks = pd.read_csv("./specs/marks_question1.csv").sort_values(by=['midterm'], ascending=True).reset_index(drop=True)
print("Head of the data")
#lets have a look at the data
print(marks.head())

# FIND THE Correlation
print(f"Correlation:\n{marks.corr()}")

#plot the grades
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(marks["midterm"], marks["final"], s=50, c="black")
ax.set_xlabel("Midterm Grade")
ax.set_ylabel("Final Grade")

print(f"Saving Plot...")

ax.figure.savefig('./output/marks.png')

print(f"Done!")
print("="*20,"\n")


print("*"*5, "Question 1 - 2\n", "*"*5)
#FIT A SIMPLE LINEAR REGRESSION LINE
reg = LinearRegression()
mt = marks["midterm"].values.reshape(-1, 1)
fn = marks["final"].values.reshape(-1, 1)
print("Fitting Linear Regression Model...")
reg.fit(mt, fn)
print(f"Done!")
b0 = reg.intercept_.item()
b1 = reg.coef_.item()
print(f"Formula for model: Y = {b0} + {b1}*X")

print("="*20, "\n")
#PREDICT THE VALUE!
print("*"*5, "Question 1 - 3\n", "*"*5)
print("Predicting value for 86")
pred = reg.predict(np.array([[86]])).item()
print(f"Done!")
print(f"Predicted value for 86 is {pred}", "\n")
print("="*20)

# Create plot
#I also added the fitted Line to a plot
print("*"*5, "I also added the fitted Line to a plot", "*"*5)
ln_x = np.array([(i / 10) for i in range(200, 1100)])
ln_y = np.array([reg.predict(np.array([[i]])).item() for i in ln_x])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(ln_x, ln_y, s=1, c="blue")
ax.scatter(marks["midterm"], marks["final"], s=50, c="black")
ax.set_xlabel("Midterm Grade")
ax.set_ylabel("Final Grade")
ax.figure.savefig('./output/marks_fitted.png')

#I also added the fitted Line to a plot with the residuals
print("*"*5, "I also added the fitted Line to a plot with the residuals", "*"*5)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(len(marks)):
    x_coords = (marks.iloc[i][0],marks.iloc[i][0])
    y_coords = (marks.iloc[i][1],reg.predict(np.array([[marks.iloc[i][0]]])).item())
    ax.plot(x_coords, y_coords, 'r--')
ax.scatter(ln_x, ln_y, s=1, c="blue")
ax.scatter(marks["midterm"], marks["final"], s=50, c="black")
ax.set_xlabel("Midterm Grade")
ax.set_ylabel("Final Grade")
ax.figure.savefig('./output/residuals.png')


# In[20]:


## PART 2 QUESTIONS!
print("="*40)
print(f"Question 2")
print("="*40, "\n")

print("*"*5, "Question 2 - 1\n", "*"*5)
print("Filtering out TID...")
b = pd.read_csv("./specs/borrower_question2.csv")
tid = b.pop("TID")
print(f"Done!")
print("="*20,"\n")


"""
We need to quickly preprocess the data tgo fit it to a decision tree so I simple just use a label encoder
the reason I use a label encoder is because they values don't have an order and the decision tree wont
bias smaller or larger labels.
"""
y = b.pop("DefaultedBorrower")
le_ho = preprocessing.LabelEncoder()
le_ms = preprocessing.LabelEncoder()
le_ho.fit(list(b['HomeOwner'].unique()))
le_ms.fit(list(b['MaritalStatus'].unique()))
b["HomeOwner"] = le_ho.transform(b["HomeOwner"])
b['MaritalStatus'] = le_ms.transform(b['MaritalStatus'])



print("*"*5, "Question 2 - 2\n", "*"*5)
s_p = './output/tree_high.png'
min_i = 0.5
print(f"Creating Decision Tree (min-impurity={min_i})...")
dtree1 = DecisionTreeClassifier(criterion="entropy", min_impurity_decrease=min_i, random_state=42)
dtree1.fit(b, y)
print(f"Done!")
print(f" Number of leaves: {dtree1.get_n_leaves()}")
print(f"Saving Tree Image ({s_p})...")
plt.figure(figsize=(12,8))
plot_tree(dtree1, feature_names=b.columns,
                   class_names=list(y.unique()),
                   filled=True)
plt.savefig(s_p, format='png', bbox_inches = "tight")
print(f"Done!")
print("="*20,"\n")

print("*"*5, "Question 2 - 3\n", "*"*5)
s_p = './output/tree_low.png'
min_i = 0.1
print(f"Creating Decision Tree (min-impurity={min_i})...")
dtree2 = DecisionTreeClassifier(criterion="entropy", min_impurity_decrease=min_i, random_state=42)
dtree2.fit(b, y)
print(f"Done!")
print(f" Number of leaves: {dtree2.get_n_leaves()}")
print(f"Saving Tree Image ({s_p})...")
plt.figure(figsize=(20,40))
plot_tree(dtree2, feature_names=b.columns,
                   class_names=list(y.unique()),
                   filled=True)
plt.savefig(s_p, format='png', bbox_inches = "tight")
print(f"Done!")
print("="*20,"\n")


print("*"*5, "Question 2 - 3\n", "*"*5)
print(f"Tree 1 importance of features: {dtree1.feature_importances_} ")
print(f"Tree 2 importance of features: {dtree2.feature_importances_} ")

print("="*20,"\n")


# In[ ]:




