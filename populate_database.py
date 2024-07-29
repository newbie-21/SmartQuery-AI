import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import time
from langchain_community.vectorstores import Chroma

CHROMA_PATH = r"E:\Abdullah\Pycharm Projects\Conversation Bot Llama\chroma"
DATA_PATH = r"E:\Abdullah\Pycharm Projects\Conversation Bot Llama\Data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    print("Starting to load documents...")
    start_time = time.time()
    documents = document_loader.load()
    end_time = time.time()
    print(f"Loaded {len(documents)} documents in {end_time - start_time:.2f} seconds.")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    print("Starting to add documents to Chroma...")
    embedding_function = get_embedding_function()
    persist_directory = CHROMA_PATH
    db = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=persist_directory)

    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"Calculated IDs for {len(chunks_with_ids)} chunks.")

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        print(f"New chunk IDs: {new_chunk_ids[:10]}...")

        batch_size = 50  # Adjust the batch size as needed
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = new_chunk_ids[i:i + batch_size]
            print(f"Adding batch {i // batch_size + 1} of {len(new_chunks) // batch_size + 1}...")
            db.add_documents(batch, ids=batch_ids)
            print(f"Batch {i // batch_size + 1} added.")

        print("All documents added. Database changes are automatically persisted.")
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()