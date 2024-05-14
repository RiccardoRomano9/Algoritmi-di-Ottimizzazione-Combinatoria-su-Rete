import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Definizione dell'algoritmo di TABU Search per il clustering
def tabu_search_clustering(data, num_iterations, tabu_size, labels_kmeans, kmeans_inertia, kmeans_centroids):
    # Inizializzazione con l'algoritmo k-means
    best_solution = labels_kmeans
    best_inertia = kmeans_inertia
    centroids = kmeans_centroids

    # Inizializzazione della lista TABU
    tabu_list = []

    for _ in range(num_iterations):
        # Generazione di una nuova soluzione vicina
        new_solution = best_solution.copy()
        i, j = np.random.choice(range(len(data)), size=2, replace=False)
        new_solution[i] = new_solution[j]
        new_solution[j] = best_solution[i]

        # Calcolo dei centroidi per la nuova soluzione
        new_centroids = np.array([np.mean(data[new_solution == label], axis=0) for label in range(kmeans.n_clusters)])

        # Calcolo dell'inertia per la nuova soluzione
        new_inertia = np.sum((data[new_solution] - new_centroids[new_solution]) ** 2)

        # Aggiornamento della soluzione migliore se la nuova soluzione è migliore
        if new_inertia < best_inertia and not any(np.array_equal(new_solution, tabu_solution) for tabu_solution in tabu_list):
            best_solution = new_solution
            best_inertia = new_inertia
            centroids = new_centroids

        # Aggiunta della nuova soluzione alla lista TABU
        tabu_list.append(new_solution)

        # Rimozione delle soluzioni più vecchie dalla lista TABU
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution, best_inertia, centroids


# Generazione del dataset Moon
X, ground_truth_labels = make_moons(n_samples=2000, noise=0.1)

# Inizializzazione dei centroidi con K-means
kmeans = KMeans(n_clusters=2)  # Solo due cluster poiché il DataSet Moon non ne ha di maggiori
kmeans.fit(X)
labels_kmeans = kmeans.predict(X)
centroids_kmeans = kmeans.cluster_centers_

# Configurazione dei parametri per la meta-euristica Tabu Search
num_iterations = 5000
tabu_size = 100

# Calcolo dell'inertia per K-means
inertia_kmeans = np.sum((X[labels_kmeans] - centroids_kmeans[labels_kmeans]) ** 2)

# Esecuzione dell'algoritmo Tabu Search
best_solution, best_inertia, centroids_tabu_search = tabu_search_clustering(X, num_iterations, tabu_size, labels_kmeans, inertia_kmeans, centroids_kmeans)


# Calcolo dell'F1 score per K-means
f1_kmeans = f1_score(ground_truth_labels, labels_kmeans, average='macro')

# Calcolo dell'F1 score per Tabu Search
f1_tabu_search = f1_score(ground_truth_labels, best_solution, average='macro')

# Plotting dei risultati
fig, axs = plt.subplots(1, 4, figsize=(20, 10))


# K-means
axs[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
axs[0].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('K-means (Inertia: {:.2f}, F1 Score: {:.2f})'.format(inertia_kmeans, f1_kmeans))

# Tabu Search
axs[1].scatter(X[:, 0], X[:, 1], c=best_solution, cmap='viridis')
axs[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', s=100, linewidths=2)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('Tabu Search (Inertia: {:.2f}, F1 Score: {:.2f})'.format(best_inertia, f1_tabu_search))

# Ground Truth
axs[2].scatter(X[:, 0], X[:, 1], c=ground_truth_labels, cmap='viridis')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].set_title('Ground Truth')

# Inertia comparison
axs[3].bar(['K-means', 'Tabu Search'], [inertia_kmeans, best_inertia])
axs[3].set_xlabel('Algorithm')
axs[3].set_ylabel('Intertia')
axs[3].set_title('Inertia Comparison')

plt.tight_layout()
plt.show()