import os
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    """
    Obtiene la función de embedding, configurando la URL base 
    desde las variables de entorno.
    """
    
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    print(f"🔄 Conectando a Ollama en: {ollama_base_url}") 

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=ollama_base_url 
    )
    return embeddings