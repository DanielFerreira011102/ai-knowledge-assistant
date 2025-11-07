from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_pdf(file_path: Path) -> List[Document]:
    """
    Loads a single PDF file and converts each page into a LangChain Document.
    
    The PyPDFLoader automatically processes the PDF, splitting it into separate
    documents where each page becomes a single LangChain Document object. The
    metadata for each document will contain the source file path and the page number.

    :param file_path: Path to the PDF file.
    :type file_path: Path
    :return: A list of LangChain Document objects, typically one per page of the PDF.
    :rtype: List[Document]
    """
    # LangChain loaders usually expect the path as a string, not a Path object,
    # so we explicitly convert it using str().
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_text(file_path: Path) -> List[Document]:
    """
    Loads a single plaintext (.txt) file into a single LangChain Document.
    
    The TextLoader reads the entire file content. For simple text files,
    we generally treat the entire file as one initial document before chunking.
    We specify UTF-8 encoding for broad compatibility with various characters.

    :param file_path: The full Path object pointing to the text file.
    :type file_path: Path
    :raises FileNotFoundError: If the text file does not exist.
    :raises UnicodeDecodeError: If the file is not correctly encoded as UTF-8.
    :return: A list containing a single Document object with the entire file's content.
    :rtype: List[Document]
    """
    # We explicitly define the encoding to avoid platform-specific issues with text files.
    loader = TextLoader(str(file_path), encoding='utf-8')
    return loader.load()


def load_document(file_path: Path) -> List[Document]:
    """
    Dispatches the loading task to the correct function based on the file extension.
    
    This function acts as a router. The design allows easy extension to new
    formats like DOCX or Markdown by simply adding a new loader function
    and another conditional branch here.

    :param file_path: The Path object of the document to be loaded.
    :type file_path: Path
    :raises ValueError: If the file extension is not supported by any available loader.
    :return: A list of Document objects resulting from the load operation.
    :rtype: List[Document]
    """
    # We get the file extension, convert it to lowercase for case-insensitive matching.
    suffix = file_path.suffix.lower()
    
    # Check for supported file types and call the corresponding loader function.
    if suffix == '.pdf':
        return load_pdf(file_path)
    
    if suffix == '.txt':
        return load_text(file_path)
    
    # If the file extension doesn't match any supported type, we immediately
    # raise a ValueError. This exception must be caught and handled by the caller,
    # ensuring unsupported files do not silently cause problems later.
    raise ValueError(f"Unsupported file format: {suffix}")


def load_documents_from_directory(directory: Path) -> List[Document]:
    """
    Scans a specified directory and loads all documents with supported extensions.
    
    This function performs file discovery and iterates through the found documents,
    calling `load_document` for each one. Unsupported files are gracefully skipped
    during the iteration.

    :param directory: The Path object of the directory containing the source files.
    :type directory: Path
    :return: A concatenated list of all Document objects successfully loaded
             from the directory. Returns an empty list if the directory is empty
             or does not exist.
    :rtype: List[Document]
    """
    # Before starting, check if the directory exists and is, in fact, a directory.
    # This prevents errors from Path().iterdir() and provides a clean exit.
    if not directory.is_dir():
        # Returning an empty list signals that no documents could be loaded.
        return []
    
    # Define which extensions are supported for the filtering step.
    supported_extensions = ['.pdf', '.txt']
    
    # We use a list comprehension to efficiently collect all files in the directory
    # that are actual files and have a supported extension.
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    # If no supported files are found, we stop here.
    if not files:
        return []
    
    # Now, we iterate through the valid files and load them.
    documents = []
    for file_path in files:
        try:
            # The load_document function returns a list of Documents (one for TXT,
            # many for PDF pages).
            docs = load_document(file_path)
            # We use extend to flatten the list of lists into a single list of Documents.
            documents.extend(docs)
        except Exception as e:
            # If any individual file fails to load (e.g., a corrupted PDF or unknown format),
            # we log the error (or print it) and continue to the next file,
            # preventing one bad file from stopping the entire ingestion process.
            print(f"Warning: Failed to load {file_path.name}. Error: {e}")
            continue
    
    return documents


def get_document_stats(documents: List[Document]) -> dict:
    """
    Calculates and returns key statistics for the loaded documents.
    
    This is an important debugging and logging step, confirming what data
    was successfully loaded before proceeding to the computationally expensive
    chunking and embedding steps.

    :param documents: A list of Document objects loaded from the disk.
    :type documents: List[Document]
    :return: A dictionary containing the total number of documents (pages/files),
             the total character count, and a list of unique source file names.
    :rtype: dict
    """
    # Handle the empty case immediately to prevent division by zero or iteration errors.
    if not documents:
        return {
            "total_documents": 0,
            "total_characters": 0,
            "sources": []
        }
    
    # We use a set to automatically handle unique source file names.
    sources = set()
    total_chars = 0
    
    for doc in documents:
        # We try to get the 'source' from the metadata, defaulting to 'unknown' if missing.
        # This metadata is automatically added by LangChain's loaders.
        sources.add(doc.metadata.get('source', 'unknown'))
        # The core content is in the 'page_content' attribute.
        total_chars += len(doc.page_content)
    
    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "sources": sorted(sources)
    }