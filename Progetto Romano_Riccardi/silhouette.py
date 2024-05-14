import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score


# Definizione dell'algoritmo di TABU Search per il clustering
def tabu_search_clustering(data, num_iterations, tabu_size, labels_kmeans):
    # Inizializzazione con l'algoritmo k-means
    best_solution = labels_kmeans
    best_cost = calculate_dunn_index(data, best_solution)

    # Inizializzazione della lista TABU
    tabu_list = []

    for _ in range(num_iterations):
        # Generazione di una nuova soluzione vicina
        new_solution = best_solution.copy()
        i, j = np.random.choice(range(len(data)), size=2, replace=False)
        new_solution[i] = new_solution[j]
        new_solution[j] = best_solution[i]

        # Calcolo dell'indice di Dunn della nuova soluzione
        new_cost = calculate_dunn_index(data, new_solution)

        # Aggiornamento della soluzione migliore se la nuova soluzione è migliore
        if new_cost > best_cost and not any(np.array_equal(new_solution, tabu_solution) for tabu_solution in tabu_list):
            best_solution = new_solution
            best_cost = new_cost

        # Aggiunta della nuova soluzione alla lista TABU
        tabu_list.append(new_solution)

        # Rimozione delle soluzioni più vecchie dalla lista TABU
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution, best_cost


# Calcolo dell'indice di Dunn per valutare il clustering
def calculate_dunn_index(data, labels):
    cluster_centers = []
    for cluster_label in np.unique(labels):
        cluster_centers.append(np.mean(data[labels == cluster_label], axis=0))
    cluster_centers = np.array(cluster_centers)
    
    # Calcolo delle distanze minime tra i centroidi dei cluster
    min_inter_cluster_distances = pairwise_distances(cluster_centers)
    min_inter_cluster_distance = np.min(min_inter_cluster_distances[min_inter_cluster_distances > 0])
    
    # Calcolo delle distanze massime all'interno di ogni cluster
    max_intra_cluster_distances = np.array([
        np.max(pairwise_distances(data[labels == cluster_label])) for cluster_label in np.unique(labels)
    ])
    max_intra_cluster_distance = np.max(max_intra_cluster_distances)
    
    # Calcolo dell'indice di Dunn
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance
    
    return dunn_index


# Generazione del dataset Moon
X, _ = make_moons(n_samples=2000, noise=0.03, random_state=0)

# Inizializzazione dei centroidi con K-means
kmeans = KMeans(n_clusters=2)  # Solo due cluster poiché il dataset Moon non ne ha di maggiori
kmeans.fit(X)
labels_kmeans = kmeans.predict(X)
centroids_kmeans = kmeans.cluster_centers_

# Configurazione dei parametri per la meta-euristica Tabu Search
num_iterations = 5000
tabu_size = 100

# Esecuzione dell'algoritmo Tabu Search
best_solution, best_cost = tabu_search_clustering(X, num_iterations, tabu_size, labels_kmeans)

# Calcolo dell'indice di Dunn per K-means
dunn_kmeans = calculate_dunn_index(X, labels_kmeans)

# Calcolo dell'indice di Dunn per Tabu Search
dunn_tabu_search = best_cost

# Plotting dei risultati
fig, axs = plt.subplots(1, 3, figsize=(15, 8))

# K-means
axs[0].scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap='viridis')
axs[0].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Clustering K-means sul dataset "Moon" (Indice di Dunn: {:.2f})'.format(dunn_kmeans))

# Tabu Search
axs[1].scatter(X[:, 0], X[:, 1], c=best_solution, cmap='viridis')
axs[1].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('Clustering Tabu Search sul dataset "Moon" (Indice di Dunn: {:.2f})'.format(dunn_tabu_search))

# Ground Trouth
axs[2].scatter(X[:, 0], X[:, 1], c=ground_truth_labels, cmap='viridis')
axs[2].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='*', color='red', s=100, linewidths=2)
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].set_title('Ground truth')

plt.tight_layout()
plt.show()
