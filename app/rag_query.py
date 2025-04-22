import numpy as np
from pymilvus import Collection, utility, connections
from llm_interface import generate_answer  # Import LLM generator

from colorama import Fore, Style, init
init(autoreset=True)  # For colored output

# Setup connection to Milvus
def connect_milvus():
    connections.connect("default", host="localhost", port="19530")

# Check if collection exists
def check_collection_exists(collection_name):
    if utility.has_collection(collection_name):
        print(Fore.GREEN + f"✅ Collection '{collection_name}' exists.")
        return True
    else:
        print(Fore.RED + f"❌ Collection '{collection_name}' does not exist.")
        return False

# Retrieve logs from Milvus
def retrieve_logs(query_vector, collection_name="logs_with_timestamp_label", top_k=5):
    connect_milvus()

    if not check_collection_exists(collection_name):
        return []

    collection = Collection(name=collection_name)

    search_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "nlist": 100},
        limit=top_k,
        output_fields=["log_content"]
    )

    logs = []
    for result in search_results[0]:
        log_content = result.entity.get('log_content')
        logs.append(log_content)

    return logs

# ✨ Main script
if __name__ == "__main__":
    # Ask the user to type their question
    user_query = input(Fore.CYAN + "\n📝 Enter your query about the logs: ")

    # For now, simulate a query embedding
    query_vector = np.random.rand(768).tolist()

    retrieved_logs = retrieve_logs(query_vector)

    if not retrieved_logs:
        print(Fore.RED + "\n⚠️ No logs found. Cannot generate an answer.")
    else:
        print(Fore.YELLOW + "\n🔍 Retrieved Logs:\n")
        for idx, log in enumerate(retrieved_logs, 1):
            print(f"{idx}. {log}")

        context = "\n".join(retrieved_logs)

        print(Fore.MAGENTA + "\n🤖 Generating an answer based on the logs...\n")
        answer = generate_answer(context, user_query)

        print(Fore.GREEN + "\n✅ Final Answer:\n")
        print(Style.BRIGHT + answer)
