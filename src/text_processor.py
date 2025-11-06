from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Initializes and configures the text splitter for optimal RAG performance.
    
    We use the RecursiveCharacterTextSplitter because it's the smartest available.
    It attempts to split the text on a defined hierarchy of separators, ensuring
    that paragraphs, then sentences, then words are kept together before resorting
    to character splitting. This is vital for maintaining semantic coherence within
    each chunk.

    :param chunk_size: The target maximum size for each text chunk (in characters).
    :type chunk_size: int
    :param chunk_overlap: The size of the shared text between consecutive chunks (in characters).
    :type chunk_overlap: int
    :return: A configured instance of the recursive text splitter.
    :rtype: RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # The separators are ordered by preference. It tries to split on double newlines
        # first (paragraph break), then single newlines (line break), then spaces (word break).
        separators=["\n\n", "\n", " ", ""]
    )


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    Executes the chunking process on a list of loaded documents.
    
    When splitting a document, the splitter ensures that the metadata from the
    original document (like 'source' and 'page number') is carried over to all
    of its resulting chunks. This is how the RAG system maintains traceability
    to the source material for citation.

    :param documents: A list of Document objects (pages/files) to be split.
    :type documents: List[Document]
    :param chunk_size: The target chunk size used to initialize the splitter.
    :type chunk_size: int
    :param chunk_overlap: The chunk overlap size used to initialize the splitter.
    :type chunk_overlap: int
    :return: A list of Document objects, now representing the smaller, processed chunks.
    :rtype: List[Document]
    """
    # A quick check for an empty list prevents unnecessary object creation.
    if not documents:
        return []
    
    # Instantiate the splitter with the parameters defined in the settings.
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # LangChain's split_documents handles iterating, splitting, and metadata transfer.
    chunks = splitter.split_documents(documents)
    
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Provides statistical metrics for the resulting chunks.
    
    Analyzing the chunk statistics helps validate that the chosen `chunk_size`
    and `chunk_overlap` parameters are appropriate for the source data. We look
    for an average size close to the target and a small minimum size, which
    can indicate "orphan" chunks that might be too small to be useful.

    :param chunks: A list of chunked Document objects.
    :type chunks: List[Document]
    :return: A dictionary containing the total number of chunks and size metrics.
    :rtype: dict
    """
    # Return zero statistics for an empty list.
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    # Calculate the character length for every chunk.
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        # We use floating-point division to get a precise average size.
        "avg_chunk_size": sum(sizes) / len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes)
    }


def preview_chunk(chunk: Document) -> dict:
    """
    Generates a concise preview of a single chunk for inspection.
    
    This function is useful in the CLI to show the user what the chunking
    process has created, confirming that the text content and source metadata
    were preserved correctly.

    :param chunk: A single Document object representing a chunk.
    :type chunk: Document
    :return: A dictionary with the chunk's length, the first 200 characters
             of content, and its associated metadata.
    :rtype: dict
    """
    preview_length = 200
    content = chunk.page_content
    
    # If the content is longer than the preview length, we truncate it and add "..."
    # to indicate continuation. Otherwise, we return the full content.
    preview_text = content[:preview_length] + "..." if len(content) > preview_length else content
    
    return {
        "length": len(content),
        "preview": preview_text,
        "metadata": chunk.metadata
    }