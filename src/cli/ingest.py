import sys
from pathlib import Path

from config.settings import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME
)
from src.document_loader import load_documents_from_directory, get_document_stats
from src.text_processor import split_documents, get_chunk_stats, preview_chunk
from src.embedding_store import (
    create_embeddings,
    test_embedding_model,
    create_vector_store,
    get_vector_store_stats
)

def print_header():
    """Prints a clear, descriptive header for the CLI application."""
    print("=" * 70)
    print("📚 DOCUMENT INGESTION PIPELINE")
    print("=" * 70)
    print()


def print_section(title: str):
    """Prints a standardized, visible section header during the process."""
    print(f"\n{'─' * 70}")
    print(f"🔹 {title}")
    print(f"{'─' * 70}")


def validate_documents_directory():
    """
    Performs critical checks to ensure the source data directory is ready.
    
    This prevents the pipeline from starting if no documents are available,
    saving the user from a pointless execution.

    :return: True if the directory exists and contains supported files, False otherwise.
    :rtype: bool
    """
    if not DOCUMENTS_DIR.exists():
        print(f"❌ Directory not found: {DOCUMENTS_DIR}")
        print(f"Please create the directory before running: mkdir {DOCUMENTS_DIR}")
        return False
    
    # We use glob patterns to quickly check for the presence of supported files.
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    txt_files = list(DOCUMENTS_DIR.glob("*.txt"))
    
    if not pdf_files and not txt_files:
        print(f"❌ No PDF or TXT files found in: {DOCUMENTS_DIR}")
        print("Please add documents to the directory and try again")
        return False
    
    return True


def main():
    """
    The main execution function for the ingestion pipeline.
    
    We wrap the entire process in a high-level try/except block to ensure
    that any unexpected error (like a failed network connection to Ollama)
    is caught and presented to the user gracefully, rather than causing a
    full, confusing traceback.
    """
    print_header()
    
    try:
        # ========================================================================
        # STEP 1: Validate setup
        # ========================================================================
        print_section("Step 1: Validating Setup and Configuration")
        
        if not validate_documents_directory():
            # If the validation fails, we exit the program immediately.
            sys.exit(1)
        
        print(f"✅ Documents directory configured: {DOCUMENTS_DIR}")
        print(f"✅ Vector DB persistence directory: {VECTOR_DB_DIR}")
        print(f"✅ Target Embedding Model: {EMBEDDING_MODEL}")
        
        # ========================================================================
        # STEP 2: Load documents from disk
        # ========================================================================
        print_section("Step 2: Loading Documents from Source Directory")
        
        # This function handles the file I/O, converting PDFs and TXT files
        # into a list of LangChain Document objects (one per page or file).
        documents = load_documents_from_directory(DOCUMENTS_DIR)
        
        if not documents:
            print("❌ No documents were successfully loaded")
            sys.exit(1)
        
        stats = get_document_stats(documents)
        print(f"✅ Successfully loaded {stats['total_documents']} documents (pages/files)")
        print(f"   Total raw character count: {stats['total_characters']:,}")
        print("   Sources loaded:")
        for source in stats['sources']:
            # We display just the filename for cleaner output.
            print(f"   - {Path(source).name}")
        
        # ========================================================================
        # STEP 3: Split documents into chunks
        # ========================================================================
        print_section("Step 3: Splitting Documents into Optimal Chunks")
        
        print(f"Chunking with parameters: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
        
        # The splitting process breaks the larger documents into smaller, overlapping
        # fragments ready for embedding.
        chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        
        chunk_stats = get_chunk_stats(chunks)
        print(f"\n✅ Created {chunk_stats['total_chunks']} searchable chunks")
        # We round the average size for readability in the console.
        print(f"   Average chunk size: {round(chunk_stats['avg_chunk_size'])} characters")
        print(f"   Size range: {chunk_stats['min_chunk_size']}-{chunk_stats['max_chunk_size']} characters")
        
        # Show a sample chunk to visualize the result of the chunking strategy.
        if chunks:
            print("\n📋 Sample chunk preview:")
            preview = preview_chunk(chunks[0])
            print(f"   Length: {preview['length']} characters")
            print(f"   Preview: {preview['preview']}")
            # We ensure the source metadata is correctly displayed.
            print(f"   Source: {Path(preview['metadata'].get('source', 'unknown')).name}")
        
        # ========================================================================
        # STEP 4: Initialize embedding model
        # ========================================================================
        print_section("Step 4: Initializing and Testing Embedding Model")
        
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Ollama URL: {OLLAMA_BASE_URL}")
        
        # This function establishes the connection to the local Ollama service.
        embeddings = create_embeddings(EMBEDDING_MODEL, OLLAMA_BASE_URL)
        
        # Test the model and connection before proceeding.
        print("\nTesting Ollama connection and model loading...")
        dimension = test_embedding_model(embeddings)
        print(f"✅ Model loaded successfully from Ollama")
        print(f"   Embedding dimension confirmed: {dimension}")
        
        # ========================================================================
        # STEP 5: Create vector store
        # ========================================================================
        print_section("Step 5: Generating Embeddings and Storing in ChromaDB")
        
        print(f"Processing {len(chunks)} chunks for embedding...")
        print("⏳ This step can take a minute (Ollama generates vectors on CPU)...")
        
        # This is the most resource-intensive step: converting every chunk into a vector.
        vector_store = create_vector_store(
            documents=chunks,
            embeddings=embeddings,
            persist_directory=VECTOR_DB_DIR,
            collection_name=COLLECTION_NAME
        )
        
        store_stats = get_vector_store_stats(vector_store)
        print(f"\n✅ Vector store created and persisted successfully")
        print(f"   Total vectors indexed: {store_stats['total_vectors']}")
        print(f"   Collection name: {store_stats['collection_name']}")
        print(f"   Saved to persistence folder: {VECTOR_DB_DIR}")
        
        # ========================================================================
        # Done!
        # ========================================================================
        print_section("Ingestion Complete")
        
        print("✅ All documents are indexed and ready for retrieval queries")
        print("Next: Run a query script to test the RAG system")
        print()

    except Exception as e:
        # A final, robust error handler for all unexpected issues.
        print("\n" + "=" * 70)
        print("❌ CRITICAL INGESTION FAILURE")
        print("=" * 70)
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        print("Please check that the Ollama service is running and the embedding model is installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()