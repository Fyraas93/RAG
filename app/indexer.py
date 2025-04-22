from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from embeddings import embed
import json
import time
import pandas as pd

# Connexion à Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Charger les logs depuis le fichier JSON
with open("../data/anomalies.json", "r", encoding="utf-8") as f:
    list_of_logs = json.load(f)
print(f"📦 {len(list_of_logs)} logs chargés depuis data/anomalies.json")

# Création d'un DataFrame pour faciliter les manipulations
logs_df = pd.DataFrame(list_of_logs)

# Convertir les timestamps (string → int64 en millisecondes)
logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], format='mixed')
logs_df["timestamp"] = logs_df["timestamp"].astype("int64") // 10**6  # millisecondes

# Définir le schéma pour la collection
log_id = FieldSchema(name="log_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
content = FieldSchema(name="log_content", dtype=DataType.VARCHAR, max_length=1000)
timestamp = FieldSchema(name="timestamp", dtype=DataType.INT64)
target_label = FieldSchema(name="target_label", dtype=DataType.VARCHAR, max_length=64)

schema = CollectionSchema(
    fields=[log_id, embedding, content, timestamp, target_label],
    description="Logs Vector DB"
)

collection_name = "logs_with_timestamp_label"

# Vérifier si la collection existe
try:
    collection = Collection(name=collection_name)
    print(f"✅ Collection '{collection_name}' existante trouvée.")
    if not collection.indexes:
        print("📌 Aucun index trouvé, création d'un nouvel index...")
        collection.create_index(field_name="embedding", index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_PQ",
            "params": {"nlist": 1024, "m": 16, "nbits": 8}
        })
    else:
        print("📚 Index déjà présent.")
except Exception as e:
    print(f"⚠️ Collection '{collection_name}' non trouvée. Création...")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_PQ",
        "params": {"nlist": 1024, "m": 16, "nbits": 8}
    })

# Pause le temps que l’index se construise
time.sleep(5)

# Charger la collection en mémoire
collection.load()

# Embedding des logs
log_texts = logs_df["log_message"].tolist()
vectors = embed(log_texts)
print(f"🧠 Exemple d'embedding : {vectors[0][:5]}...")  # Affiche un bout de l'embedding

# Données à insérer dans l’ordre des champs
entities = [
    vectors.tolist(),                           # embedding
    log_texts,                                  # log_content
    logs_df["timestamp"].tolist(),              # timestamp (int64)
    logs_df["target_label"].tolist()            # target_label
]

# Insertion
try:
    collection.insert(entities)
    collection.flush()
    print("✅ Logs indexés avec succès dans Milvus !")
except Exception as e:
    print(f"❌ Erreur lors de l'insertion des logs : {e}")
