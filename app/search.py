from pymilvus import connections, Collection
import numpy as np

# Étape 1 : Se connecter à Milvus (ajuste l'adresse si besoin)
connections.connect(
    alias="default",
    host="localhost",  # ou l'IP de ton serveur Milvus
    port="19530"
)

# Étape 2 : Se connecter à la collection
collection = Collection("logs_with_timestamp_label")

# Étape 3 : Embedding à rechercher (à remplacer par un vrai vecteur)
query_embedding = np.random.rand(1, 512).tolist()  # format attendu : List[List[float]]

# Étape 4 : Recherche
search_results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["log_content", "timestamp", "target_label"]
)

# Étape 5 : Affichage des résultats
for hits in search_results:
    for hit in hits:
        print(f"Log trouvé : {hit.entity.get('log_content')}")
        print(f"Score de distance : {hit.distance}")
        print(f"Label : {hit.entity.get('target_label')}, Timestamp : {hit.entity.get('timestamp')}")
        print("-" * 50)
