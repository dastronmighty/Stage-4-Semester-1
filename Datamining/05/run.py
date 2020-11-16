### Imports

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import pyplot as plt

import pathlib

### Imports

"""
    Function to calculate a polynomial model from an X,y and output
    the predictions of test input
"""
def poly_mod(deg, X, y, test):
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X, y)
    lin_reg = LinearRegression().fit(X_poly,y)
    preds = lin_reg.predict(poly_reg.fit_transform(test))
    return preds


### SETUP OUTPUT PATH


pathlib.Path('./output').mkdir(exist_ok=True)

# Questions

q = 1
data = pd.read_csv("./specs/markB_question.csv")
print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

q1_X = np.array(data["MCQ1"]).reshape(-1,1)
q1_y = np.array(data["final"]).reshape(-1,1)

print("\tFitting Linear Model")
reg = LinearRegression()
reg = reg.fit(q1_X, q1_y)

print("\tPredicting the final grades...")
preds = reg.predict(q1_X)
print("\tSave predictions in Data")
data["final_linear"] = preds

print("="*20)
print(f"Done!")
print("="*20)

print("*\n*\n*\n*")
print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

b0 = reg.intercept_.item()
b1 = reg.coef_.item()

print("\tModel Params:\n\t\tIntercept = {}\n\t\tSlope = {}".format(b0,b1))

print("="*20)
print(f"Done!")
print("="*20)

print("*\n*\n*\n*")
print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

q3_X = np.array(data["MCQ1"]).reshape(-1,1)
q3_y = np.array(data["final"]).reshape(-1,1)

for deg in [2, 3, 4, 8, 10]:
    print(f"\tPloynomial Model (degree {deg})...")
    print(f"\tPredicting with Model (degree {deg})...")
    preds = poly_mod(deg, q3_X, q3_y, q3_X)
    print("\tSave predictions in Data (degree {deg})")
    print("\t*"*5)
    data["final_poly"+str(deg)] = preds
print("="*20)
print(f"Done!")
print("="*20)
print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

print("\tSaving Ouput in ./output/question_mcq1.csv")
data.to_csv("./output/question_mcq1.csv")

print("="*20)
print(f"Done!")
print("="*20)

print("*\n*\n*\n*")
data = pd.read_csv("./specs/markB_question.csv")
print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

#part1
print("\t Part 1", "\t="*5)
q1_X = np.array(data["MCQ2"]).reshape(-1,1)
q1_y = np.array(data["final"]).reshape(-1,1)

print("\tFitting Linear Model")
reg2 = LinearRegression()
reg2 = reg2.fit(q1_X, q1_y)

print("\tPredicting the final grades...")
preds = reg2.predict(q1_X)
print("\tSave predictions in Data")
data["final_linear"] = preds

#part 2
print("\t Part 2", "\t="*5)
b0 = reg2.intercept_.item()
b1 = reg2.coef_.item()
print("\tModel Params:\n\t\tIntercept = {}\n\t\tSlope = {}".format(b0,b1))

#part 3
print("\t Part 3", "\t="*5)
q3_X = np.array(data["MCQ2"]).reshape(-1,1)
q3_y = np.array(data["final"]).reshape(-1,1)

for deg in [2, 3, 4, 8, 10]:
    print(f"\tPloynomial Model (degree {deg})...")
    print(f"\tPredicting with Model (degree {deg})...")
    preds = poly_mod(deg, q3_X, q3_y, q3_X)
    print("\tSave predictions in Data (degree {deg})")
    print("\t*"*5)
    data["final_poly"+str(deg)] = preds

#part 4
print("\t Part 4", "\t="*5)
print("\tSaving Ouput in ./output/question_mcq2.csv")
data.to_csv("./output/question_mcq2.csv")

print("="*20)
print(f"Done!")
print("="*20)

print("*\n*\n*\n*")

print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

data = pd.read_csv("./specs/markB_question.csv")

X = data.iloc[:, 0:2].values
y = data.iloc[:, 2:3].values

lin_reg=LinearRegression()
lin_reg.fit(X,y)

b0 = lin_reg.intercept_.item()
b1 = lin_reg.coef_[0][0]
b2 = lin_reg.coef_[0][1]
print(f"\tModel Params:\n\t\tIntercept = {b0}\n\t\tSlope1 = {b1}\n\t\tSlope2 = {b2}")

print("\tPredicting the final grades...")
preds = lin_reg.predict(X)
data["final_linear"] = preds

print("\tSaving Ouput in ./output/question_full.csv")
data.to_csv("./output/question_full.csv")

print("="*20)
print(f"Done!")
print("="*20)

fig = plt.figure(figsize=(12,8))
plt.title("Final Grade on MCQ 1")
ax = fig.add_subplot(1,1,1)
ax.grid(color='lightgrey', linestyle='-', linewidth=1)
ax.set_xlabel("MCQ1 Grade")
ax.set_ylabel("Final Grade")

data = pd.read_csv("./specs/markB_question.csv")
X = np.array(data["MCQ1"]).reshape(-1,1)
y = np.array(data["final"]).reshape(-1,1)

ax.scatter(data["MCQ1"], data["final"], s=150, c="tab:blue")
ln_x = np.array([(i / 10) for i in range(390, 970)])
ln_x_ = ln_x.reshape(-1, 1)
ln_reg_y = reg.predict(ln_x_)
ax.plot(ln_x, ln_reg_y, 'r')
for deg, col in zip([2,3,4,8,10], ['black', 'blue', 'yellow', 'green', 'orange']):
    preds = poly_mod(deg, X, y, ln_x_)
    ax.plot(ln_x, preds, col, linestyle='dashed')
ax.legend(['linear','poly (2)','poly (3)','poly (4)','poly (8)','poly (10)','Training points'], loc=4)
ax.figure.savefig('./output/question_mcq1.pdf', dpi=500)


fig = plt.figure(figsize=(12,8))
plt.title("Final Grade on MCQ 2")
ax = fig.add_subplot(1,1,1)
ax.grid(color='lightgrey', linestyle='-', linewidth=1)
ax.set_xlabel("MCQ2 Grade")
ax.set_ylabel("Final Grade")

data = pd.read_csv("./specs/markB_question.csv")
X = np.array(data["MCQ2"]).reshape(-1,1)
y = np.array(data["final"]).reshape(-1,1)

ax.scatter(data["MCQ2"], data["final"], s=150, c="tab:blue")
ln_x = np.array([(i / 10) for i in range(390, 970)])
ln_x_ = ln_x.reshape(-1, 1)
ln_reg_y = reg2.predict(ln_x_)
ax.plot(ln_x, ln_reg_y, 'r')
for deg, col in zip([2,3,4,8,10], ['black', 'blue', 'yellow', 'green', 'orange']):
    preds = poly_mod(deg, X, y, ln_x_)
    ax.plot(ln_x, preds, col, linestyle='dashed')
ax.legend(['linear','poly (2)','poly (3)','poly (4)','poly (8)','poly (10)','Training points'], loc=4)
ax.figure.savefig('./output/question_mcq2.pdf', dpi=500)
