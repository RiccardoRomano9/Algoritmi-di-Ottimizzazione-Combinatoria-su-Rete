import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from copy import deepcopy

# Definizione dell'algoritmo di TABU Search per il clustering
def tabu_search_clustering(data, k, num_iterations, tabu_size):
    # Inizializzazione con l'algoritmo k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    best_solution = kmeans.labels_

    # Inizializzazione della lista TABU
    tabu_list = []

    for _ in range(num_iterations):
        # Generazione di una nuova soluzione vicina
        new_solution = best_solution.copy()
        i, j = np.random.choice(range(len(data)), size=2, replace=False)
        new_solution[i] = new_solution[j]
        new_solution[j] = best_solution[i]

        # Aggiornamento della soluzione migliore se la nuova soluzione è migliore
        if not any(np.array_equal(new_solution, tabu_solution) for tabu_solution in tabu_list):
            best_solution = new_solution

        # Aggiunta della nuova soluzione alla lista TABU
        tabu_list.append(new_solution)

        # Rimozione delle soluzioni più vecchie dalla lista TABU
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution


# Generazione del dataset Moon
X, _ = make_moons(n_samples=2000, noise=0.5, random_state=0)

# Inizializzazione dei centroidi con K-means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels_kmeans = kmeans.predict(X)
centroids_kmeans = kmeans.cluster_centers_

# Configurazione dei parametri per l'algoritmo Tabu Search
num_iterations = 100
tabu_size = 10

# Esecuzione dell'algoritmo Tabu Search
best_solution = tabu_search_clustering(X, 2, num_iterations, tabu_size)

# Plotting dei risultati
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# K-means
axs[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis')
axs[0].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Clustering K-means sul dataset "Moon"')

# Tabu Search
axs[1].scatter(X[:, 0], X[:, 1], c=best_solution, cmap='viridis')
axs[1].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('Clustering Tabu Search sul dataset "Moon"')

plt.tight_layout()
plt.show()
