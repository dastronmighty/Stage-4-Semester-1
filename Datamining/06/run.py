#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import pathlib


# ### SETUP OUTPUT PATH

# In[2]:


pathlib.Path('./output').mkdir(exist_ok=True)


# # Global Constants

# In[3]:


RANDOM_STATE = 0
FIG_SIZE = (8, 5)
COLORS = ['tab:blue', 'tab:orange', 'tab:green', "tab:red", "tab:purple", "tab:pink", "tab:cyan"]
CENTER_COL = "black"
S = 50
SC = round(S * 0.5)
C_M = "X"
DPI = 700
legend_elems = [
    Line2D([0], [0], marker=C_M, color='w', label='Cluster Center', markerfacecolor=CENTER_COL, markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor=COLORS[0], markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Cluster 2',markerfacecolor=COLORS[1], markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Cluster 3',markerfacecolor=COLORS[2], markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Cluster 4',markerfacecolor=COLORS[3], markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Cluster 5',markerfacecolor=COLORS[4], markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Cluster 6',markerfacecolor=COLORS[5], markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Cluster 7',markerfacecolor=COLORS[6], markersize=15)
]
q = 1


# #### If in a notebook you can set this to True to see the plots inline

# In[4]:


TURN_ON_PLOT = False


# # Questions

# In[5]:


"""
=======================================
Question 1
=======================================
"""

print("="*40)
print(f"Question {q}")
print("="*40)
q+=1
q1_df = pd.read_csv("./specs/question_1.csv")

print("\tFitting K-Means")
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit(q1_df.values)

print("\tLabeling data-points")
q1_df["cluster"] = kmeans.labels_

print("\tSaving Data 'output/question_1.csv'")
q1_df.to_csv("output/question_1.csv")

print("\tPlotting...")
# PLOTS
#############################
xmax, xmin = round(q1_df["x"].max()) + 1, round(q1_df["x"].min() - 1)
xstep = round(abs((xmax - xmin)) / 5)
ymax, ymin = round(q1_df["y"].max()) + 1, round(q1_df["y"].min() - 1)
ystep = round(abs((ymax - ymin)) / 5)
X_Ticks = [i for i in range(xmin, xmax, xstep)]
Y_Ticks = [i for i in range(ymin, ymax, ystep)]
q1_df["colors"] = q1_df["cluster"].apply(lambda x: COLORS[x])
#############################
fig = plt.figure(figsize=FIG_SIZE) 
ax = fig.add_subplot(1,1,1, label="Q1")
ax.set_title("K-Means")
ax.set_xlabel("X-Coords")
ax.set_ylabel("Y-Coords")
ax.scatter(q1_df["x"], q1_df["y"], s=S, c=q1_df["colors"])
ax.set_xticks(X_Ticks)
ax.set_yticks(Y_Ticks)
ax.legend(handles=legend_elems[1:4])
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
print("Saving Plot: ./output/question_1.pdf")
ax.figure.savefig('./output/question_1.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q1C1")
ax.set_title("K-Means with centers")
ax.set_xlabel("X-Coords")
ax.set_ylabel("Y-Coords")
ax.set_xticks(X_Ticks)
ax.set_yticks(Y_Ticks)
ax.scatter(q1_df["x"], q1_df["y"], s=S, c=q1_df["colors"])
for c, center in enumerate(kmeans.cluster_centers_):
    ax.scatter(center[0], center[1], s=SC, c=CENTER_COL, marker=C_M)
    ax.annotate("Cluster_"+str(c), xy=(center[0], center[1]))
ax.legend(handles=legend_elems[0:4])
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
print("Saving Plot: ./output/question_1_complimentary1.pdf")
ax.figure.savefig('./output/question_1_complimentary1.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q1C2")
ax.set_title("K-Means with centers and lines")
ax.set_xlabel("X-Coords")
ax.set_ylabel("Y-Coords")
ax.set_xticks(X_Ticks)
ax.set_yticks(Y_Ticks)
centroids = [kmeans.cluster_centers_[:,i] for i in [0, 1]]
clusters = [q1_df[q1_df["cluster"]==i] for i in [0, 1, 2]]
for i, cluster in enumerate(clusters):
    for p in cluster.values:
        x = [centroids[0][i], p[0]]
        y = [centroids[1][i], p[1]] 
        ax.plot(x, y, p[3], linewidth=1, zorder=1)
ax.scatter(q1_df["x"], q1_df["y"], s=S, c=q1_df["colors"])
for c, center in enumerate(kmeans.cluster_centers_):
    ax.scatter(center[0], center[1], s=SC, c=CENTER_COL, zorder=2, marker=C_M)
ax.legend(handles=legend_elems[0:4])
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
print("Saving Plot: ./output/question_1_complimentary2.pdf")
ax.figure.savefig('./output/question_1_complimentary2.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################

print("="*20)
print(f"Done!")
print("="*20)
print("*\n*\n*\n*")



"""
=======================================
Question 2
=======================================
"""

print("="*40)
print(f"Question {q}")
print("="*40)
q+=1


q2_df = pd.read_csv("./specs/question_2.csv")
discard = ["NAME", "MANUF", "TYPE", "RATING"]
print(f"\tDiscarding: {discard}")
q2_df = q2_df[[c for c in q2_df.columns if c not in discard]]
columns = q2_df.columns

clusters, runs, iterations = 5, 5, 100 
print("\tFitting K-Means 1")
print(f"\t\tTarget Clusters: {clusters}\n\t\tMax. Runs: {runs}\n\t\tMax. Iterations {iterations}")
kmeans1 = KMeans(n_clusters=clusters, n_init=runs, max_iter=iterations, random_state=RANDOM_STATE)
kmeans1 = kmeans1.fit(q2_df[columns].values)
q2_df["config1"] = kmeans1.labels_

clusters, runs, iterations = 5, 100, 100 
print("\tFitting K-Means 2")
print(f"\t\tTarget Clusters: {clusters}\n\t\tMax. Runs: {runs}\n\t\tMax. Iterations {iterations}")
kmeans2 = KMeans(n_clusters=clusters, n_init=runs, max_iter=iterations, random_state=RANDOM_STATE)
kmeans2 = kmeans2.fit(q2_df[columns].values)
q2_df["config2"] = kmeans2.labels_

print("\n\tCheck if the results are similar\n")

clusters, runs, iterations = 3, 100, 100 
print("\tFitting K-Means 3")
print(f"\t\tTarget Clusters: {clusters}\n\t\tMax. Runs: {runs}\n\t\tMax. Iterations {iterations}")
kmeans3 = KMeans(n_clusters=clusters, n_init=runs, max_iter=iterations, random_state=RANDOM_STATE)
kmeans3 = kmeans3.fit(q2_df[columns].values)
q2_df["config3"] = kmeans3.labels_

print("\n\tCheck All Clustering Results\n")

print("\tSaving Data 'output/question_2.csv'")
q2_df.to_csv("output/question_2.csv")


# PLOTS
print("\tPlotting...")
#############################
pca = PCA(n_components=2)
pca.fit(q2_df[columns].values)
pca_fitted = pca.transform(q2_df[columns].values)
pca_q2_df = pd.DataFrame({"x":pca_fitted[:,0], 
                          "y": pca_fitted[:,1], 
                          "k1":q2_df["config1"],
                          "k2":q2_df["config2"],
                           "k3":q2_df["config3"]})
pca_q2_df["c1"] = pca_q2_df["k1"].apply(lambda x: COLORS[x])
pca_q2_df["c2"] = pca_q2_df["k2"].apply(lambda x: COLORS[x])
pca_q2_df["c3"] = pca_q2_df["k3"].apply(lambda x: COLORS[x])
k1_centers = pca.transform(kmeans1.cluster_centers_)
k2_centers = pca.transform(kmeans2.cluster_centers_)
k3_centers = pca.transform(kmeans3.cluster_centers_)
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q2C1")
ax.set_title("K-Means | 5 clusters | 5 runs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
ax.scatter(pca_q2_df["x"], pca_q2_df["y"], s=S, c=pca_q2_df["c1"])
ax.scatter(k1_centers[:,0], k1_centers[:,1], s=SC, c=CENTER_COL, marker=C_M)
ax.legend(handles=legend_elems[0:6])
print("Saving Plot: ./output/question_2_complimentary1.pdf")
ax.figure.savefig('./output/question_2_complimentary1.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q2C2")
ax.set_title("K-Means | 5 clusters | 100 runs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
ax.scatter(pca_q2_df["x"], pca_q2_df["y"], s=S, c=pca_q2_df["c2"])
ax.scatter(k2_centers[:,0], k2_centers[:,1], s=SC, c=CENTER_COL, marker=C_M)
ax.legend(handles=legend_elems[0:6])
print("Saving Plot: ./output/question_2_complimentary2.pdf")
ax.figure.savefig('./output/question_2_complimentary2.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q2C3")
ax.set_title("K-Means | 3 clusters | 100 runs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
ax.scatter(pca_q2_df["x"], pca_q2_df["y"], s=S, c=pca_q2_df["c3"])
ax.scatter(k3_centers[:,0], k3_centers[:,1], s=SC, c=CENTER_COL, marker=C_M)
ax.legend(handles=legend_elems[0:4])
print("Saving Plot: ./output/question_2_complimentary3.pdf")
ax.figure.savefig('./output/question_2_complimentary3.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################

print("="*20)
print(f"Done!")
print("="*20)
print("*\n*\n*\n*")



"""
=======================================
Question 3
=======================================
"""

print("="*40)
print(f"Question {q}")
print("="*40)
q+=1

q3_df = pd.read_csv("./specs/question_3.csv")

print(f"\tDiscarding ID Column")
columns = ["x", "y"]
q3_df = q3_df[columns]

clusters, runs, iterations = 7, 5, 100 
print("\tFitting K-Means")
print(f"\t\tTarget Clusters: {clusters}\n\t\tMax. Runs: {runs}\n\t\tMax. Iterations {iterations}")
q3_km1 = KMeans(n_clusters=clusters, n_init=runs, max_iter=iterations, random_state=RANDOM_STATE)
q3_km1 = q3_km1.fit(q3_df[columns].values)
print("\tK-Means Values into Data")
q3_df["kmeans"] = q3_km1.labels_
km1_colors = q3_df["kmeans"].apply(lambda x: COLORS[x])

print("\tMinMax Scaling")
scaler = MinMaxScaler()
scaler.fit(q3_df[columns].values)
trans_data = scaler.transform(q3_df[columns].values)

scan, e, m_sam = 1, 0.04, 4
print(f"\tDBSCAN {scan}...")
print(f"\t\tEpsilon: {e}\n\t\tMin Points: {m_sam}")
dbscan_cluster1 = DBSCAN(eps=e, min_samples=m_sam).fit(trans_data)
labs1 = dbscan_cluster1.labels_
print("\tDBSCAN 1 Values into Data")
q3_df["dbscan1"] = labs1

scan, e, m_sam = 2, 0.08, 4
print(f"\tDBSCAN {scan}...")
print(f"\t\tEpsilon: {e}\n\t\tMin Points: {m_sam}")
dbscan_cluster1 = DBSCAN(eps=e, min_samples=m_sam).fit(trans_data)
labs2 = dbscan_cluster1.labels_
print("\tDBSCAN 2 Values into Data")
q3_df["dbscan2"] = labs2

print("\tSaving Data 'output/question_3.csv'")
q3_df.to_csv("output/question_3.csv")

# PLOTS
print("\tPlotting...")
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q3C1")
ax.set_title("K-Means | 7 clusters | 5 runs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.scatter(q3_df["x"], q3_df["y"], s=S, c=km1_colors)
ax.scatter(q3_km1.cluster_centers_[:,0], q3_km1.cluster_centers_[:,1], s=SC, c=CENTER_COL, marker=C_M)
ax.legend(handles=legend_elems, bbox_to_anchor=(0.5, -0.2), fancybox=False, ncol=4, loc="center")
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
print("Saving Plot: ./output/question_3_1.pdf")
ax.figure.savefig('./output/question_3_1.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q3C2")

ax.set_title("DBSCAN 1")
ax.set_xlabel("X")
ax.set_ylabel("Y")

colors = list(map(lambda x: COLORS[x], labs1))
ax.scatter(trans_data[:,0], trans_data[:,1], 
           s=S, 
           c=colors)
l_elems = []
for i, c in enumerate(np.unique(colors)):
    l_elems.append(
        Line2D([0], [0], 
               marker='o', color='w', 
               label='Cluster '+str(i+1),
               markerfacecolor=np.unique(colors)[i], 
               markersize=15)
    )
ax.legend(handles=l_elems, bbox_to_anchor=(0.5, -0.2), fancybox=False, ncol=4, loc="center")
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
print("Saving Plot: ./output/question_3_2.pdf")
ax.figure.savefig('./output/question_3_2.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(1,1,1, label="Q3C3")

ax.set_title("DBSCAN 2")
ax.set_xlabel("X")
ax.set_ylabel("Y")

colors = list(map(lambda x: COLORS[x], labs2))
ax.scatter(trans_data[:,0], trans_data[:,1], 
           s=S, 
           c=colors)
l_elems = []
for i, c in enumerate(np.unique(colors)):
    l_elems.append(
        Line2D([0], [0], 
               marker='o', color='w', 
               label='Cluster '+str(i+1),
               markerfacecolor=np.unique(colors)[i], 
               markersize=15)
    )
lgd = ax.legend(handles=l_elems, bbox_to_anchor=(0.5, -0.2), fancybox=False, ncol=4, loc="center")
ax.set_axisbelow(True)
ax.yaxis.grid(color='tab:gray', linestyle='dashed')
ax.xaxis.grid(color='tab:gray', linestyle='dashed')
print("Saving Plot: ./output/question_3_3.pdf")
fig.savefig('./output/question_3_3.pdf', dpi=DPI, bbox_inches='tight')
if not TURN_ON_PLOT:
    plt.close()
#############################

