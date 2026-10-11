from ingest import process_all_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """Splitting the documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    #documents that i have split to chunks
    split_docs=text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")

    return split_docs


#chunks=split_documents(process_all_pdfs("data"))
