import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

# Definizione dell'algoritmo di TABU Search per il clustering
def tabu_search_clustering(data, num_iterations, tabu_size, labels_kmeans, ground_truth_labels):
    # Inizializzazione con l'algoritmo k-means
    best_solution = labels_kmeans
    best_cost = f1_score(ground_truth_labels, best_solution, average='weighted')

    # Inizializzazione della lista TABU
    tabu_list = []

    for _ in range(num_iterations):
        # Generazione di una nuova soluzione vicina
        new_solution = best_solution.copy()
        i, j = np.random.choice(range(len(data)), size=2, replace=False)
        new_solution[i] = new_solution[j]
        new_solution[j] = best_solution[i]

        # Calcolo dello score della nuova soluzione
        new_cost = f1_score(ground_truth_labels, new_solution, average='weighted')
        # Calcolo dei centroidi per la nuova soluzione
        new_centroids = np.array([np.mean(data[new_solution == label], axis=0) for label in range(kmeans.n_clusters)])
        # Aggiornamento della soluzione migliore se la nuova soluzione è migliore
        if new_cost > best_cost and not any(np.array_equal(new_solution, tabu_solution) for tabu_solution in tabu_list):
            best_solution = new_solution
            best_cost = new_cost
            best_centroids = new_centroids
        # Aggiunta della nuova soluzione alla lista TABU
        tabu_list.append(new_solution)

        # Rimozione delle soluzioni più vecchie dalla lista TABU
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution, best_cost, best_centroids

# Generazione del dataset Moon
X, ground_truth_labels = make_moons(n_samples=2000, noise=0.1)

# Inizializzazione dei centroidi con K-means
kmeans = KMeans(n_clusters=2)      #Solo due cluster poichè il DataSet Moon non ne ha di maggiori
kmeans.fit(X)
labels_kmeans = kmeans.predict(X)
centroids_kmeans = kmeans.cluster_centers_

# Configurazione dei parametri per la meta-euristica Tabu Search
num_iterations = 5000
tabu_size = 100

# Esecuzione dell'algoritmo Tabu Search
best_solution, best_cost, best_centroids = tabu_search_clustering(X, num_iterations, tabu_size, labels_kmeans, ground_truth_labels)

# Calcolo dell'F1-score per K-means
f1_kmeans = f1_score(ground_truth_labels, kmeans.predict(X))

# Calcolo dell'F1-score per Tabu Search
f1_tabu_search = f1_score(ground_truth_labels, best_solution)
centroids_tabu = best_centroids
# Plotting dei risultati
fig, axs = plt.subplots(1, 4, figsize=(15, 8))

# K-means
axs[0].scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap='viridis')
axs[0].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Clustering K-means (F1-score: {:.2f})'.format(f1_kmeans))

# Tabu Search
axs[1].scatter(X[:, 0], X[:, 1], c=best_solution, cmap='viridis')
axs[1].scatter(centroids_tabu[:, 0], centroids_tabu[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('Clustering Tabu Search (F1-score: {:.2f})'.format(f1_tabu_search))

# Ground Trouth
axs[2].scatter(X[:, 0], X[:, 1], c=ground_truth_labels, cmap='viridis')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].set_title('Ground trouth')

# Inertia comparison
axs[3].bar(['K-means', 'Tabu Search'], [f1_kmeans, f1_tabu_search])
axs[3].set_xlabel('Algorithm')
axs[3].set_ylabel('Intertia')
axs[3].set_title('F1-Score Comparison')

plt.tight_layout()
plt.show()