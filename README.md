# Soluzione a Problema di Clustering mediante Tabu Search

Questo repository contiene un progetto di Ricerca Operativa che affronta un problema di clustering utilizzando l'algoritmo K-Means migliorato con la meta-euristica del Tabu Search.

## Idea del Progetto

L'idea del progetto è partita dal presupposto che, affrontando un problema di clustering su un dataset generico (utilizzeremo il dataset Moon), sia possibile migliorare l'efficacia dell'algoritmo K-Means utilizzando il Tabu Search. Il clustering è un tipo di problema di apprendimento non supervisionato che coinvolge la suddivisione di un insieme di dati in gruppi omogenei in base alle loro caratteristiche comuni.

## K-Means

Il K-Means è un algoritmo di clustering ampiamente utilizzato che cerca di suddividere un insieme di dati non etichettati in un numero predefinito di cluster. L'algoritmo funziona iterativamente per assegnare ciascun punto dati a uno dei cluster, cercando di minimizzare la somma delle distanze quadrate tra i punti e i centroidi dei cluster.

## Tabu Search

Il Tabu Search è una meta-euristica utilizzata per risolvere problemi di ottimizzazione combinatoria. A differenza di altri algoritmi di ricerca locale, il Tabu Search permette di esplorare anche soluzioni che potrebbero temporaneamente peggiorare il valore della funzione obiettivo, utilizzando una lista tabù per evitare di ritornare a soluzioni precedentemente visitate.

## Score

L'F1-score è una misura comune dell'accuratezza dei modelli di classificazione che combina precisione e richiamo in un unico valore. Varia tra 0 e 1, dove 1 indica la massima precisione e richiamo, mentre 0 indica la peggiore performance. È particolarmente utile con dataset sbilanciati e fornisce una valutazione accurata del modello quando sia la precisione che il richiamo sono importanti.

## Risultato Esemplificativo

![Gif D'Esempio](https://github.com/RiccardoRomano9/Soluzione-Problema-di-Clustering-con-Tabu-Search/blob/main/Progetto%20Romano_Riccardi/tabu_search_clustering_F1.gif)
