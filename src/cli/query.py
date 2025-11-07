import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Add parent directory to path so we can import our modules.
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    LLM_MODEL,
    LLM_TEMPERATURE
)
from src.embedding_store import (
    create_embeddings,
    load_vector_store,
    similarity_search
)
from src.llm import create_llm, generate_answer


def print_header():
    """Prints a clear, descriptive header for the CLI application."""
    print("=" * 70)
    print("💬 KNOWLEDGE ASSISTANT - Retrieval Augmented Generation (RAG) Query Interface")
    print("=" * 70)
    print()


def print_section(title: str):
    """Prints a standardized, visible section header during the process."""
    print(f"\n{'─' * 70}")
    print(f"🔹 {title}")
    print(f"{'─' * 70}")


def print_separator():
    """Prints a simple, visual separator for better readability between query steps."""
    print(f"\n{'─' * 70}\n")


def validate_vector_store():
    """
    Performs a critical check to ensure the vector database persistence
    directory exists before attempting to load the store.

    This prevents the pipeline from starting if the ingestion step has
    not been successfully completed.

    :return: True if the directory is found, False otherwise.
    :rtype: bool
    """
    if not VECTOR_DB_DIR.exists():
        print(f"❌ Vector database not found: {VECTOR_DB_DIR}")
        print(f"👉 Please run the document ingestion pipeline first: python src/cli/ingest.py")
        return False
    
    return True


def display_retrieved_chunks(chunks):
    """
    Displays a formatted preview of the retrieved chunks to the user.

    This helps the user build intuition about the retrieval step and can be
    valuable for debugging and understanding what context the LLM is receiving
    to form its answer.

    :param chunks: A list of LangChain Document objects retrieved from the vector store.
    :type chunks: List[Documents]
    """
    print(f"📚 Retrieved {len(chunks)} relevant chunks:\n")
    
    for i, chunk in enumerate(chunks, 1):
        # Extract the metadata for each chunk (source document and page number).
        source = chunk.metadata.get('source', 'unknown')
        source_name = Path(source).name
        page = chunk.metadata.get('page', 'N/A')
        
        # Display a truncated preview of each chunk.
        preview_length = 150
        content = chunk.page_content
        preview = content[:preview_length] + "..." if len(content) > preview_length else content
        
        print(f"Chunk {i}:")
        print(f"  Source: {source_name} (page {page})")
        print(f"  Preview: {preview}")
        print()


def format_context(chunks) -> str:
    """
    Formats a list of retrieved document chunks into a single string to be
    used as the 'context' block in the LLM prompt.

    This string is prepended to the user's question before being passed to the LLM. 
    By numbering and labeling each chunk, we instruct the LLM on how to use the 
    information and reference its sources accurately in the final answer.

    :param chunks: A list of LangChain Document objects.
    :type chunks: List[Documents]
    :return: A formatted string containing all chunk content and source metadata.
    :rtype: str
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        # Extract the metadata for each chunk (source document and page number).
        source = chunk.metadata.get('source', 'unknown')
        source_name = Path(source).name
        page = chunk.metadata.get('page', 'N/A')
        
        # Append the chunk content along with its metadata to the context string.
        context_parts.append(
            f"[Chunk {i} from {source_name}, page {page}]\n{chunk.page_content}"
        )
    
    return "\n\n".join(context_parts)


def create_prompt(question: str, context: str) -> str:
    """
    Generates the final, structured prompt payload for the LLM.
    Assembles the complete, structured instruction (prompt) to be sent to the LLM.
    
    This prompt integrates all the necessary components to successfully execute the 
    RAG flow, including:
    1. A clear system instruction (defining the persona and task).
    2. The retrieved document context (the external knowledge base).
    3. The original question posed by the user.
    4. Explicit constraints and guidelines (e.g., ONLY use the context).

    :param question: The question asked by the user.
    :type question: str
    :param context: The pre-formatted string containing all relevant document chunks.
    :type context: str
    :return: The final prompt string, ready to be sent to the LLM.
    :rtype: str
    """
    prompt = f"""You are a helpful assistant that answers questions based on provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain the answer, say so clearly (e.g., 'I cannot find the answer in the provided documents.')
- Be concise but complete
- Reference which chunk(s) your answer comes from when possible (e.g., '... (Chunk 3)')

Answer:"""
    
    return prompt


def create_prompt(question: str, context: str) -> str:
    """
    Generates the final, structured prompt payload for the LLM.
    Assembles the complete, structured instruction (prompt) to be sent to the LLM.
    
    This prompt integrates all the necessary components to successfully execute the 
    RAG flow, including:
    1. A clear system instruction (defining the persona and task).
    2. The retrieved document context (the external knowledge base).
    3. The original question posed by the user.
    4. Explicit constraints and guidelines (e.g., ONLY use the context).

    :param question: The question asked by the user.
    :type question: str
    :param context: The pre-formatted string containing all relevant document chunks.
    :type context: str
    :return: The final prompt string, ready to be sent to the LLM.
    :rtype: str
    """
    
    # Define a clear system instruction and formatting
    SYSTEM_INSTRUCTION = (
        "You are an expert, fact-checking assistant. Your sole purpose is to "
        "synthesize a precise and accurate answer based **only** on the "
        "provided 'CONTEXT'. You must strictly adhere to the instructions."
    )

    # Using triple-backticks (```) or XML tags in the prompt is a common,
    # and highly recommended, practice to clearly delineate the context and
    # instructions from the main task for the LLM.
    prompt = f"""
### System Role
{SYSTEM_INSTRUCTION}

### CONTEXT
```{context}```

### User Question
{question}

### Instructions and Constraints
1. You MUST use the information found in the 'CONTEXT' to formulate your answer. Do not use any external knowledge.
2. After providing the answer, cite the source document name and/or page number for each distinct piece of information used, enclosed in parentheses (e.g., '...is 12 (Source: document_a.pdf, p. 5)'). If source information is not available in the context, skip this step.
3. Be comprehensive in answering the user's question, but do not add unnecessary preamble or filler. Get straight to the point.
4. If the necessary information to answer the question is NOT present in the 'CONTEXT', you must state clearly and politely: "I cannot find a definitive answer to your question in the provided documents."

### Answer:
"""
    return prompt


def interactive_mode(vector_store, embeddings, llm):
    """
    Runs the RAG query pipeline in a continuous interactive loop.

    This mode allows the user to ask multiple questions against the indexed
    documents without restarting the script, providing a true conversational
    experience (though each query is stateless).

    :param vector_store: The loaded vector store instance (e.g., ChromaDB).
    :param embeddings: The initialized embedding model for query embedding.
    :param llm: The initialized Large Language Model instance.
    """
    # NOTE: The print_section for this mode is moved to main() for consistency.
    print("Type 'exit' or 'quit' to stop the conversation.\n")
    
    while True:
        # Prompt the user for a question.
        question = input("❓ Your question: ").strip()
        
        # Check if the user wants to exit.
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty questions.
        if not question:
            continue
                
        # ========================================================================
        # RAG Step 1: Retrieval
        # ========================================================================
        print(f"\n🔍 Searching for the top {TOP_K_RESULTS} relevant document chunks...")
        # Use the embedding model to find relevant chunks in the vector store.
        chunks = similarity_search(vector_store, question, k=TOP_K_RESULTS)
        
        if not chunks:
            print("❌ No relevant information was found in your indexed documents.")
            print_separator()
            continue
        
        print(f"✅ Found {len(chunks)} relevant chunks.")
        
        # Display the retrieved chunks (context) for transparency and debugging.
        display_retrieved_chunks(chunks)
        
        # ========================================================================
        # RAG Step 2: Generation Setup
        # ========================================================================
        # Prepare the retrieved documents for injection into the prompt.
        context = format_context(chunks)
        # Build the final prompt with the system instructions, context, and question.
        prompt = create_prompt(question, context)
        
        # ========================================================================
        # RAG Step 3: Generation
        # ========================================================================
        print("💭 Generating answer from context using the LLM...")
        # Send the prompt to the LLM so it can generate an answer.
        answer = generate_answer(llm, prompt)
        
        print("\n🤖 Answer:")
        print(answer)
        
        print_separator()


def single_question_mode(vector_store, embeddings, llm, question: str):
    """
    Executes the RAG query pipeline for a single question and then exits.

    This mode is useful for non-interactive scripting or when the user provides
    the question directly as a command-line argument.

    :param vector_store: The loaded vector store instance (e.g., ChromaDB).
    :param embeddings: The initialized embedding model.
    :param llm: The initialized Large Language Model instance.
    :param question: The single question string to be answered.
    :type question: str
    """
    # NOTE: The print_section for this mode is moved to main() for consistency.
    print(f"Question: {question}\n")
    
    # ========================================================================
    # RAG Step 1: Retrieval
    # ========================================================================
    print(f"🔍 Searching for the top {TOP_K_RESULTS} relevant document chunks...")
    chunks = similarity_search(vector_store, question, k=TOP_K_RESULTS)
    
    if not chunks:
        print("❌ No relevant information was found in your indexed documents.")
        return
    
    print(f"✅ Found {len(chunks)} relevant chunks.")
    display_retrieved_chunks(chunks)
    
    # ========================================================================
    # RAG Step 2 & 3: Generation
    # ========================================================================
    context = format_context(chunks)
    prompt = create_prompt(question, context)
    
    print("💭 Generating answer from context using the LLM...")
    answer = generate_answer(llm, prompt)
    
    print("\n🤖 RAG Answer:")
    print(answer)
    print()


def main():
    """
    The main execution function for the Retrieval Augmented Generation (RAG) pipeline.

    It handles setup validation, loading the necessary components (vector store,
    embedding model, LLM), and then delegates execution to either the
    interactive or single-question mode based on command-line arguments.
    """
    print_header()
    
    try:
        # ========================================================================
        # STEP 1: Validate Setup
        # ========================================================================
        print_section("Step 1: Validating Vector Store Setup")
        
        if not validate_vector_store():
            sys.exit(1)

        print(f"✅ Vector database directory found: {VECTOR_DB_DIR}")
        
        # ========================================================================
        # STEP 2: Load Vector Store and Embedding Model
        # ========================================================================
        print_section("Step 2: Loading Vector Store and Embedding Model")
        
        # Initialize the same embedding model used during ingestion.
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Ollama URL: {OLLAMA_BASE_URL}")
        embeddings = create_embeddings(EMBEDDING_MODEL, OLLAMA_BASE_URL)
        
        # Attempt to load the persisted vector store.
        print("\n📂 Attempting to load vector store from persistence directory...")
        vector_store = load_vector_store(embeddings, VECTOR_DB_DIR, COLLECTION_NAME)
        
        if vector_store is None:
            # This handles unexpected errors during the load process.
            print("❌ Failed to load vector store.")
            print("👉 Please ensure the collection and embeddings model match the ingestion run.")
            sys.exit(1)
        
        print(f"✅ Loaded vector store from {VECTOR_DB_DIR}")
        print(f"   Total vectors indexed: {vector_store._collection.count()}")
        
        # ========================================================================
        # STEP 3: Initialize Language Model (LLM)
        # ========================================================================
        # This print_section was missing in the original and is added for consistency
        print_section("Step 3: Initializing Language Model (LLM)")
        print(f"Model: {LLM_MODEL}")
        print(f"Ollama URL: {OLLAMA_BASE_URL}") # Added for consistency
        
        # Establish the connection to the Ollama LLM service
        llm = create_llm(LLM_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE)
        
        print(f"✅ LLM Model initialized: {LLM_MODEL}")
        print("✅ RAG system components are ready for querying.")
        
        # ========================================================================
        # STEP 4: Enter Query Mode
        # ========================================================================
        
        # Check if a question was provided as a command-line argument.
        if len(sys.argv) > 1:
            # Single question mode: Execute the RAG pipeline for a single question.
            question = " ".join(sys.argv[1:])
            print_section("Step 4: Single Question Query Mode")
            single_question_mode(vector_store, embeddings, llm, question)
        else:
            # Interactive mode: Start the continuous Q&A loop.
            print_section("Step 4: Interactive Query Mode")
            interactive_mode(vector_store, embeddings, llm)

    except Exception as e:
        # A final, robust error handler for all unexpected issues.
        print("\n" + "=" * 70)
        print("❌ CRITICAL QUERY FAILURE")
        print("=" * 70)
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        print("Please check that the Ollama service is running and the models are available.")
        sys.exit(1)


if __name__ == "__main__":
    main()