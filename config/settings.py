from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# The root directory of the project. 
# All other paths are defined relative to this directory so the structure 
# remains consistent no matter where the code is run.
PROJECT_ROOT = Path(__file__).parent.parent

# This defines the location where we store the source material (PDFs, TXT files)
# that we want the RAG system to learn from. The ingestion script will scan this
# folder to find new documents to index.
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# This is the directory where ChromaDB will store its internal files. 
# ChromaDB is a "persistent" vector database, meaning that after we generate 
# the embeddings and save them, we can shut down the application and load 
# the database back up later without having to re-index all the documents again.
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"

# ============================================================================
# EMBEDDING MODEL
# ============================================================================

# The embedding model is responsible for converting human-readable text into
# numerical vectors (lists of numbers). The quality of these vectors directly
# impacts how well the retrieval system works.
#
# We are using "nomic-embed-text" because it is a powerful open-source model
# optimized for retrieval and is small enough to run effectively on a local CPU
# using Ollama. It is important to use the EXACT same embedding model for both
# indexing (creating the database) and querying (searching the database).
# Using different models will create incompatible vector spaces, and the
# similarity search will fail completely.
EMBEDDING_MODEL = "nomic-embed-text"

# This is the address where the local Ollama server is running.
# The `OllamaEmbeddings` and `ChatOllama` classes use this URL to communicate
# with the service over HTTP. Ollama acts as a wrapper that makes it easy to run
# various models locally.
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================================
# TEXT CHUNKING PARAMETERS
# ============================================================================

# Since LLMs can't digest an entire document (like a book or a long report) 
# all at once, we need to chop up these larger texts into smaller, more
# manageable pieces called "chunks".
#
# The chunk size sets the absolute maximum number of characters allowed in one
# of these pieces before we send it off to the embedding model or the LLM.
# If this value is too small (e.g., less than 500 characters), the resulting
# pieces might not have enough surrounding text for the LLM to understand what
# the document is about or to answer our questions correctly.
#
# If it is too large (e.g., more than 2000 characters), we risk polluting the
# model's limited memory (context window) with unnecessary text which wastes
# tokens and can confuse the model. A sweet spot is usually found between 1000
# and 1500 characters for most documents.
CHUNK_SIZE = 1000

# This setting determines how many characters should be shared between two
# consecutive chunks. We use this overlap to avoid losing important context
# that might get cut off exactly at the boundary between two chunks.
#
# For instance, if a sentence spans the end of chunk A and the start of chunk B,
# a small overlap makes sure the entire sentence, or at least a large part of
# it, appears in both chunks. A good rule of thumb is to set the overlap to
# about 10% or 20% of the chunk size.
CHUNK_OVERLAP = 200

# ============================================================================
# LANGUAGE MODEL (for answering questions)
# ============================================================================

# This is the LLM that processes the information retrieved from our vector
# database, places it into context, and then uses that context to generate the
# final answer for the user. 
#
# We are currently using "llama3.2" because it is a highly capable model that is
# also efficient enough to run locally using the Ollama service on a standard
# CPU.
LLM_MODEL = "llama3.2"

# The temperature parameter determines how much the model "thinks outside the
# box." A very low temperature (close to 0.0) tells the model to play it safe
# and always pick the most probable next word, making its answers very rigid
# and predictable.
#
# A high temperature (e.g., 0.7 or above) encourages the model to take more
# risks, leading to more diverse and creative responses albeit with a higher
# chance of making things up (hallucinations).
#
# For a system that answers questions based on facts (like RAG), we want the
# model to be precise, not imaginative. Therefore, it is generally better to
# keep the temperature low (between 0.0 and 0.3) to ensure the answers are
# reliable and grounded in the provided context.
LLM_TEMPERATURE = 0.1

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================

# This is the "k" parameter for k-Nearest Neighbors search.
# It defines how many of the most relevant chunks (vectors) the retriever will
# pull from the vector database (ChromaDB) and pass to the LLM after a user
# asks a question.
#
# If this number is too low (e.g., k=1), the LLM may not have enough context to
# generate a good answer. If it's too high (e.g., k=20), the LLM may be
# overwhelmed with information, some of which may be irrelevant or contradictory,
# leading to confusion or "hallucination." It can also slow down the response
# time, as the LLM has to process more data.
TOP_K_RESULTS = 4

# ============================================================================
# CHROMADB SETTINGS
# ============================================================================

# A collection in ChromaDB is similar to a table in a relational database. It
# is a distinct set of vectors (embeddings) that share the same schema and
# purpose. Here, we define the name of the collection that will store our
# document embeddings.
COLLECTION_NAME = "knowledge_base"

# This is the distance metric ChromaDB uses to measure the similarity between
# the query vector and the document vectors stored in the database.
#
# For text embeddings, "cosine" similarity is often preferred over "l2"
# (Euclidean Distance) and "ip" (Inner Product) because it compares the
# direction of the vectors (their semantic meaning) rather than their
# magnitude, which can vary depending on the length of the text.
DISTANCE_METRIC = "cosine"
