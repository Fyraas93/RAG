from transformers import pipeline

# Load a small T5 model for lightweight generation
rag_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small"
)

def generate_answer(context, user_query):
    """
    Generate an answer using the retrieved logs as context + user question.
    """
    prompt = f"Given the following logs:\n{context}\n\nAnswer the user's question:\n{user_query}"
    output = rag_pipeline(prompt, max_new_tokens=200, do_sample=True)
    return output[0]['generated_text']
