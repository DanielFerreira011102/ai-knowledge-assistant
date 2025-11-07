from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# We import the DISTANCE_METRIC from the settings to enforce the correct
# similarity calculation when creating the Chroma collection.
from config.settings import DISTANCE_METRIC


def create_embeddings(model_name: str, base_url: str) -> OllamaEmbeddings:
    """
    Initializes the connection to the Ollama server for embedding generation.
    
    This sets up the client that communicates with the local Ollama API,
    telling it which specific embedding model (`nomic-embed-text` in our case)
    to use for transforming text into high-dimensional vectors.

    :param model_name: The name of the embedding model to load from Ollama.
    :type model_name: str
    :param base_url: The network URL for the running Ollama server.
    :type base_url: str
    :return: A configured OllamaEmbeddings client instance.
    :rtype: OllamaEmbeddings
    """
    # The OllamaEmbeddings class handles the API calls and model management.
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url
    )


def test_embedding_model(embeddings: OllamaEmbeddings) -> int:
    """
    Tests the connection and functionality of the embedding model.
    
    By embedding a simple test string, we verify that the Ollama service is
    reachable and correctly serving the requested model. The dimension size is
    a confirmation that the model loaded is the expected one (e.g., 768 for nomic-embed-text).

    :param embeddings: The configured embedding model client.
    :type embeddings: OllamaEmbeddings
    :raises requests.exceptions.ConnectionError: If the Ollama server is not running or unreachable.
    :return: The dimension size (length) of the generated vector.
    :rtype: int
    """
    # We embed a short query and check the length of the resulting list of numbers.
    test_vector = embeddings.embed_query("test")
    return len(test_vector)


def create_vector_store(
    documents: List[Document],
    embeddings: OllamaEmbeddings,
    persist_directory: Path,
    collection_name: str
) -> Chroma:
    """
    Creates a new vector store and persists it to the disk.
    
    This is the core indexing step. ChromaDB iterates over all documents,
    calls the embedding model to generate a vector for each chunk, and then
    stores the vector, the original text, and the metadata together.

    :param documents: A list of Document objects (the chunks) to embed and store.
    :type documents: List[Document]
    :param embeddings: The configured embedding model client.
    :type embeddings: OllamaEmbeddings
    :param persist_directory: The location where the Chroma database files will be saved.
    :type persist_directory: Path
    :param collection_name: The name assigned to the collection within the database.
    :type collection_name: str
    :raises ValueError: If the input document list is empty, as we cannot create an empty store.
    :return: The newly created and indexed ChromaDB vector store instance.
    :rtype: Chroma
    """
    if not documents:
        raise ValueError("Cannot create vector store from an empty document list")
    
    # We define the collection metadata to enforce the desired distance metric.
    # If we don't do this, Chroma defaults to L2 (Euclidean distance),
    # which contradicts our configuration for cosine similarity.
    # By the way, "hnsw" refers to the Hierarchical Navigable Small Worlds algorithm,
    # which is a graph-based approximate nearest neighbor search technique used in many
    # vector databases, including
    collection_metadata = {"hnsw:space": DISTANCE_METRIC} 

    # Chroma.from_documents is a convenience method that handles both creating the
    # database and adding all documents in a single step.
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        collection_metadata=collection_metadata # Enforce the correct distance metric
    )
    
    return vector_store


def load_vector_store(
    embeddings: OllamaEmbeddings,
    persist_directory: Path,
    collection_name: str
) -> Optional[Chroma]:
    """
    Loads an existing vector store from the disk.
    
    This function allows the application to restart without having to re-index all
    the data, making the query process much faster. The critical requirement is that
    the same `embeddings` object (using the same model) must be provided, or
    the loaded vectors will be incompatible with new query vectors.

    :param embeddings: The configured embedding model client, matching the one used
                       during the store's creation.
    :type embeddings: OllamaEmbeddings
    :param persist_directory: The location where the database is saved.
    :type persist_directory: Path
    :param collection_name: The name of the collection to load.
    :type collection_name: str
    :return: The loaded ChromaDB vector store, or None if the database is not found
             or is empty.
    :rtype: Optional[Chroma]
    """
    # We check for the directory's existence first.
    if not persist_directory.exists():
        return None
    
    try:
        # We try to initialize the Chroma client by pointing it to the persistence folder.
        # This only loads the database structure, not the vectors yet.
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # We perform a quick count check to ensure the collection actually contains vectors
        # and did not load an empty or corrupted structure.
        if vector_store._collection.count() == 0:
            return None
        
        return vector_store
        
    except Exception as e:
        # A broader exception catch here handles potential issues like file permission errors
        # or a partially corrupted database, preventing a hard crash.
        print(f"Warning: Failed to load vector store from {persist_directory}. Error: {e}")
        return None


def get_vector_store_stats(vector_store: Chroma) -> dict:
    """
    Retrieves and reports key statistics about the indexed collection.
    
    This confirms to the user how many chunks were successfully converted into vectors
    and are available for retrieval.

    :param vector_store: The ChromaDB vector store instance.
    :type vector_store: Chroma
    :return: A dictionary containing the total number of vectors and the collection name.
    :rtype: dict
    """
    # The internal collection object provides access to methods like 'count'.
    collection = vector_store._collection
    
    return {
        "total_vectors": collection.count(),
        "collection_name": collection.name
    }


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Executes the actual similarity search (Retrieval) operation.
    
    This is the heart of the RAG system's retrieval step. The query string is
    first converted into an embedding vector, and then Chroma rapidly searches
    for the `k` most geometrically similar vectors in the database based on the
    configured distance metric (cosine).

    :param vector_store: The ChromaDB vector store instance to search against.
    :type vector_store: Chroma
    :param query: The user's search query string.
    :type query: str
    :param k: The number of top similar documents (chunks) to return.
    :type k: int
    :return: A list of the most semantically similar Document objects found.
    :rtype: List[Document]
    """
    # The `similarity_search` method handles the query embedding, the search,
    # and the retrieval of the original chunk text and metadata.
    return vector_store.similarity_search(query, k=k)