import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calculate_inertia(X, labels, centers):
    inertia = 0
    for i, x in enumerate(X):
        centroid = centers[labels[i]]
        inertia += np.linalg.norm(x - centroid) ** 2
    return inertia

def tabu_search_kmeans(X, k, max_iterations, tabu_size):
    n_samples, n_features = X.shape
    
    km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    km.fit(X)
    best_labels = km.labels_
    best_centers = km.cluster_centers_
    best_inertia = calculate_inertia(X, best_labels, best_centers)
    
    tabu_list = []
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        candidate_labels = np.copy(best_labels)
        candidate_centers = np.copy(best_centers)
        
        move_point = np.random.randint(n_samples)
        move_cluster = np.random.randint(k)
        
        candidate_labels[move_point] = move_cluster
        
        for cluster in range(k):
            mask = (candidate_labels == cluster)
            if np.any(mask):
                candidate_centers[cluster] = np.mean(X[mask], axis=0)
        
        candidate_inertia = calculate_inertia(X, candidate_labels, candidate_centers)
        
        if candidate_inertia < best_inertia or iteration % tabu_size == 0:
            best_labels = candidate_labels
            best_centers = candidate_centers
            best_inertia = candidate_inertia
        
        tabu_list.append((move_point, move_cluster))
        
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
    
    return best_labels, best_centers

# Funzione per plottare i centroidi
def plot_centroids(centers):
    plt.scatter(
        centers[:, 0], centers[:, 1],
        s=50, marker='o',
        c='black', edgecolor='black',
        label='centroids')

# Funzione per calcolare l'indice di silhouette
def calculate_silhouette(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg

# Esempio di utilizzo
dt = pd.read_csv('xclara.csv', header=None)
X = dt.iloc[:, [0,1]].values

k = 5
max_iterations = 300
tabu_size = 100

# K-means
km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
km.fit(X)
km_labels = km.labels_
km_silhouette = calculate_silhouette(X, km_labels)
print("Indice di silhouette per K-means:", km_silhouette)

# Tabu Search
labels, centers = tabu_search_kmeans(X, k, max_iterations, tabu_size)
tabu_labels = labels
tabu_silhouette = calculate_silhouette(X, tabu_labels)
print("Indice di silhouette per Tabu Search:", tabu_silhouette)

# Ground Truth
ground_truth_labels = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1]
ground_truth_silhouette = calculate_silhouette(X, ground_truth_labels)
print("Indice di silhouette per Ground Truth:", ground_truth_silhouette)

# Plot del grafico per K-means
plt.figure(figsize=(12, 5))
plot_centroids(km.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering Results')
plt.show()

# Plot del grafico per Tabu Search
plt.figure(figsize=(12, 5))
plot_centroids(centers)
plt.scatter(X[labels==0, 0], X[labels==0, 1], s=30, c='red', label='Cluster 1')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=30, c='blue', label='Cluster 2')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=30, c='green', label='Cluster 3')
plt.scatter(X[labels==3, 0], X[labels==3, 1], s=30, c='cyan', label='Cluster 4')
plt.scatter(X[labels==4, 0], X[labels==4, 1], s=30, c='magenta', label='Cluster 5')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Tabu Search Clustering Results')
plt.legend()
plt.show()

# Plot del grafico per Ground Truth
plt.figure(figsize=(12, 5))
plot_centroids(km.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=ground_truth_labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Ground Truth Clustering Results')
plt.show()
