import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# reading all the pdfs on the directory
def process_all_pdfs(pdf_directory):
    """Processing all files in my directory"""
    all_documents=[]
    pdf_dir=Path(pdf_directory)

    pdf_files=list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    # looping through the files 
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}")
        try:
            loader=PyMuPDFLoader(str(pdf_file))
            documents=loader.load() 

            # adding metadata

            for doc in documents:
                doc.metadata["source_file"]=str(pdf_file)
                doc.metadata["file_type"]='pdf'

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents

process_all_pdfs("data")  