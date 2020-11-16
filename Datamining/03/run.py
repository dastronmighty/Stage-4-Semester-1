import pandas as pd
import pathlib
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import KBinsDiscretizer

pathlib.Path('./output').mkdir(exist_ok=True)

def expand_df(dfx):
    cols = list(dfx.columns)
    f_c = cols.pop(0)
    df1 = pd.get_dummies(dfx[f_c], prefix=f_c).reset_index()
    for c in cols:
        df1 = df1.merge(pd.get_dummies(dfx[c], prefix=c).reset_index(), left_on='index', right_on='index')
    df1.pop("index")
    return df1

def print_rule(rule):
    ante = list(rule["antecedents"][0])
    cons = list(rule["consequents"][0])
    conf = rule["confidence"][0]
    print(f"Intesting rule:\n\t{ante} => {cons}\n\t(confidence: {conf}%)")

reqd = ["support","confidence","antecedents","consequents"]

print("="*40)
print(f"Question 1")
print("="*40)

print(f"Question 1 - task 1\n\tRemoving Count column...")
df = pd.read_csv("./specs/gpa_question1.csv")
df.pop("count")
print("Done!")
print("")

print(f"Question 1 - task 2\n\tRunning Apriori Algorithm...")
adf = expand_df(df)
apriori_df = apriori(adf, min_support=0.15, use_colnames=True, verbose=True)
print("Done!")
print("")

print(f"Question 1 - task 3\n\tSaving output...")
apriori_df.to_csv('./output/question1_out_apriori.csv', index=False)
print("Done!")
print("")

print(f"Question 1 - task 4\n\tGenerating Association Rules 0.9 conf.")
rule9 = association_rules(apriori_df, metric="confidence", min_threshold=0.9)[reqd]
print("Done!")
print("")

print(f"Question 1 - task 5\n\tSaving Association Rules 0.9...")
rule9.to_csv('./output/question1_out_rules9.csv', index=False)
print("Done!")
print("")

print(f"Question 1 - task 6\n\tGenerating Association Rules 0.7 conf.")
rule7 = association_rules(apriori_df, metric="confidence", min_threshold=0.7)[reqd]
print("Done!")
print("")

print(f"Question 1 - task 7\n\tSaving Association Rules 0.7...")
rule7.to_csv('./output/question1_out_rules7.csv', index=False)
print("Done!")
print("")

print("="*40)
print(f"Question 1 Finished")
print("="*40)

print("\n" * 3)
print("="*40)
print(f"Question 2")
print("="*40)
print(f"Question 1 - task 0\n\tReading csv...")
df = pd.read_csv("./specs/bank_data_question2.csv")
print("Done!")
print("")

print(f"Question 1 - task 1\n\tRemoving id...")
df.pop("id")
print("Done!")
print("")

print(f"Question 1 - task 2\n\tDiscretize numberic values...")
dfd = df.copy()
dfd["age"] = pd.cut(dfd["age"], 3, precision=0, duplicates="drop")
dfd["income"] = pd.cut(dfd["income"], 3, precision=0, duplicates="drop")
dfd["children"] = pd.cut(dfd["children"], 3, precision=0, duplicates="drop")
print("Done!")
print("")

print(f"Question 1 - task 3\n\tRunning Fpgrowth Algorithm...")
fpg_df = expand_df(dfd)
fpgrowth_res = fpgrowth(fpg_df, min_support=0.2, use_colnames=True)
print("Done!")
print("")

print(f"Question 1 - task 4\n\tSaving uotput...")
fpgrowth_res.to_csv('./output/question2_out_fpgrowth.csv', index=False)
print("Done!")
print("")

print(f"Question 1 - task 5\n\tObtaining at least 10 association rules")
rules10 = association_rules(fpgrowth_res, metric="confidence", min_threshold=0.79)
print("Done!")
print("")

print(f"Question 1 - task 6\n\tSaving the association rules")
rules10.to_csv('./output/question2_out_rules.csv', index=False)
print("Done!")
print("")

print(f"Question 1 - task 7\n\tInteresting Rules")
sorted_rules_m = rules10.sort_values(by='confidence', ascending=False).reset_index(drop=True)[reqd]
print_rule(sorted_rules_m[0:1].reset_index())
print_rule(sorted_rules_m[2:3].reset_index())
print(f"Done")


print("="*40)
print(f"Question 2 Finished")
print("="*40)
