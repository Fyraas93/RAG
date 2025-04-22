from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")  # Modèle E5

def embed(texts):
    # E5 exige des préfixes ("passage:" pour les documents)
    return model.encode(["passage: " + text for text in texts], convert_to_tensor=True)


print("Embedding model loaded")