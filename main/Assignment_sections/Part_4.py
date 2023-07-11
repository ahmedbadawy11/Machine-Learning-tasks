


import matplotlib.pyplot as plt
from modeling.classification_models import ClassificationModel
from visualization.plot import Draw_plot_part_4
from sklearn.cluster import KMeans
from minisom import MiniSom
import  pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# part (2) _4

# def clustering_Kmeans(X_train, y_train, X_test,df):
#     legitimate_data = df[df['Ligitimacy'] == 1]
#     modeling = ClassificationModel(X_train, y_train, X_test)
#     num_clusters = [8, 12, 16, 20, 32]
#     legitimate_members = []
#
#     for n in num_clusters:
#         labels=modeling.Kmeans(n,legitimate_data[['Latitude', 'Longitude']])
#
#         unique_labels = set(labels)
#         legitimate_cluster_members = sum((labels[i] in unique_labels) for i, label in enumerate(labels))
#         legitimate_members.append(legitimate_cluster_members)
#
#     # Plot the results
#     Draw_plot_part_4(num_clusters,legitimate_members,"KMEANS")
    # plt.plot(num_clusters, legitimate_members, marker='o')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Total Number of Legitimate-only Members')
    # plt.title('Number of Clusters vs Total Legitimate-only Members')
    # plt.show()


def clustering_Kmeans(X_train, y_train, X_test,df,train_dataset):
    legitimate_data = train_dataset[train_dataset['Ligitimacy'] == 1]
      # Select only latitude and longitude features for clustering
    X = legitimate_data[['Latitude', 'Longitude']]

    # Initialize lists to store cluster sizes and legitimate-only member counts
    cluster_sizes = []
    legitimate_member_counts = []

    # Loop over different number of clusters
    for n_clusters in [8, 12, 16, 20, 32]:
        # Create the K-means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Fit the model to the data
        kmeans.fit(X)

        # Predict the cluster labels for each sample
        labels = kmeans.predict(X)

        # Count the number of samples in each cluster
        cluster_counts = pd.Series(labels).value_counts().sort_index()

        # Filter clusters with only legitimate samples
        legitimate_clusters = cluster_counts[cluster_counts.index.isin(legitimate_data.index)]

        # Store the cluster sizes and legitimate-only member counts
        cluster_sizes.append(len(cluster_counts))
        legitimate_member_counts.append(sum(legitimate_clusters))

    # Plot the number of clusters vs. total number of legitimate-only members
    Draw_plot_part_4(cluster_sizes, legitimate_member_counts, "KMEANS")
    # plt.figure(figsize=(8, 6))
    # plt.plot(cluster_sizes, legitimate_member_counts, marker='o', linestyle='-', color='b')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Total Legitimate-only Members')
    # plt.title('Number of Clusters vs. Total Legitimate-only Members')
    # plt.grid(True)
    # plt.show()

def clustering_SOFM(X_train, y_train, X_test,df,train_dataset):
    legitimate_data = train_dataset[train_dataset['Ligitimacy'] == 1]
    X = legitimate_data[['Latitude', 'Longitude']].values

    # Initialize lists to store cluster sizes and legitimate-only member counts
    cluster_sizes = []
    legitimate_member_counts = []

    # Loop over different number of clusters
    for n_clusters in [8, 12, 16, 20, 32]:
        # Create the SOFM model
        som = MiniSom(n_clusters, 1, X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)

        # Initialize the weights using random samples
        som.random_weights_init(X)

        # Train the model
        som.train_random(X, num_iteration=100)

        # Get the closest cluster for each sample
        labels = [som.winner(x)[0] for x in X]

        # Count the number of samples in each cluster
        cluster_counts = pd.Series(labels).value_counts().sort_index()

        # Filter clusters with only legitimate samples
        legitimate_clusters = cluster_counts[cluster_counts.index.isin(legitimate_data.index)]

        # Store the cluster sizes and legitimate-only member counts
        cluster_sizes.append(len(cluster_counts))
        legitimate_member_counts.append(sum(legitimate_clusters))

    # Plot the number of clusters vs. total number of legitimate-only members
    Draw_plot_part_4(cluster_sizes,legitimate_member_counts,"SOFM")

    # plt.figure(figsize=(8, 6))
    # plt.plot(cluster_sizes, legitimate_member_counts, marker='o', linestyle='-', color='b')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Total Legitimate-only Members')
    # plt.title('Number of Clusters vs. Total Legitimate-only Members (SOFM)')
    # plt.grid(True)
    # plt.show()



def clustering_DBSCAN(X_train, y_train, X_test,df,train_dataset):
    # find DBSCAN optimal eps and min-samples
    Our_data = train_dataset[['Latitude', 'Longitude', 'Ligitimacy']]
    X = Our_data.iloc[:, :-1].values
    y_true = Our_data.iloc[:, -1].astype(int).values
    num_clusters = [8, 12, 16, 20, 32]
    epsList, msList, accList, clusterList = list(), list(), list(), list()
    # tqdm
    for eps in [0.01,0.02,0.04,0.2 ,0.5, 0.7 ,1]:
        for ms in range(2, 10):
            model = DBSCAN(eps=eps, min_samples=ms)
            predLabels = model.fit_predict(X)
            labels = model.labels_
            num_clusters_pred = len(np.unique(predLabels))-1
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and n_clusters in num_clusters:
                # score = silhouette_score(X, predLabels, random_state=0)
                epsList.append(eps)
                msList.append(ms)
                # accList.append(score)
                clusterList.append(num_clusters_pred)

    epsList, msList, accList, clusterList = np.array(epsList), np.array(msList), np.array(accList), np.array(
        clusterList)
# _________________________________________________________       DBSCAn_____________________________
#     # Select only latitude and longitude features for clustering
#     legitimate_data = train_dataset[train_dataset['Ligitimacy'] == 1]
#
#     X = legitimate_data[['Latitude', 'Longitude']].values
#
#     # Initialize lists to store cluster sizes and legitimate-only member counts
#     cluster_sizes = []
#     legitimate_member_counts = []
#
#     # Loop over different parameters
#     for midpoint, epsilon in [(5, 0.1), (10, 0.15), (15, 0.2), (20, 0.25), (30, 0.3)]:
#         # Create the DBSCAN model
#         dbscan = DBSCAN(eps=epsilon, min_samples=midpoint, metric='euclidean')
#
#         # Fit the model to the data
#         dbscan.fit(X)
#
#         # Get the predicted cluster labels
#         labels = dbscan.labels_
#
#         # Count the number of clusters (excluding noise points)
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#
#         # Count the number of samples in each cluster
#         cluster_counts = pd.Series(labels).value_counts().sort_index()
#
#         # Filter clusters with only legitimate samples
#         legitimate_clusters = cluster_counts[cluster_counts.index.isin(legitimate_data.index)]
#
#         # Store the cluster sizes and legitimate-only member counts
#         cluster_sizes.append(n_clusters)
#         legitimate_member_counts.append(sum(legitimate_clusters))
#
#     # Plot the number of clusters vs. total number of legitimate-only members
#     Draw_plot_part_4(cluster_sizes,legitimate_member_counts,"DBSCAN")
#
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(cluster_sizes, legitimate_member_counts, marker='o', linestyle='-', color='b')
#     # plt.xlabel('Number of Clusters')
#     # plt.ylabel('Total Legitimate-only Members')
#     # plt.title('Number of Clusters vs. Total Legitimate-only Members (DBSCAN)')
#     # plt.grid(True)
#     # plt.show()
