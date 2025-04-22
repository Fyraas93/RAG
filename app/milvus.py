from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 1. Connexion à Milvus
connections.connect("default", host="localhost", port="19530")

# 2. Définir les champs
fields = [
    FieldSchema(
        name="log_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=768
    ),
    FieldSchema(
        name="log_content",
        dtype=DataType.VARCHAR,
        max_length=1000
    ),
    FieldSchema(
        name="timestamp",
        dtype=DataType.INT64,
        description="UNIX timestamp of the log"
    ),
    FieldSchema(
        name="target_label",
        dtype=DataType.VARCHAR,
        max_length=100
    )
]

# 3. Schéma de la collection
schema = CollectionSchema(fields=fields, description="Log collection with timestamp and target label")

# 4. Création de la collection si elle n'existe pas
collection_name = "logs_with_timestamp_label"

if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' créée avec succès.")
else:
    print(f"ℹ️ La collection '{collection_name}' existe déjà.")

# Indexer les embeddings
collection.create_index(field_name="embedding", index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 100}})
