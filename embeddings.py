from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embeddings_batch(list_of_json_strings):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dims
        input=list_of_json_strings
    )
    return [item.embedding for item in response.data]
