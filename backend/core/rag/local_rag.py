import os
import chromadb
from sentence_transformers import SentenceTransformer
import docx
import fitz
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import argparse
import torch

class LocalRAG:
    def __init__(self, rag_folder="~/Documents/RAG/", db_path="~/Documents/RAG/.local_rag_db"):
        self.rag_folder = os.path.expanduser(rag_folder)
        self.chroma_path = os.path.expanduser(db_path)
        os.makedirs(self.rag_folder, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection("local_rag")
        # maybe use distiluse-base-multilingual-cased-v2 for multilingual support
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.model.device}")


    def extract_text_from_pdf(self, file_path):
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)

    def extract_text_from_docx(self, file_path):
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def extract_text_from_epub(self, file_path):
        book = epub.read_epub(file_path)
        texts = []
        for item in book.get_items():
            # Check if the item is an HTML document
            if item.get_type() == "application/xhtml+xml":
                soup = BeautifulSoup(item.get_content(), "html.parser")
                texts.append(soup.get_text())
        return "\n".join(texts)
    
    def extract_text_from_html(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            return soup.get_text()
    
    def load_document(self, file_path):
        filepath = file_path.lower()
        if filepath.endswith(".pdf"):
            text = self.extract_text_from_pdf(file_path)
        elif filepath.endswith(".docx"):
            text = self.extract_text_from_docx(file_path)
        elif filepath.endswith(".epub"):
            text = self.extract_text_from_epub(file_path)
        elif filepath.endswith(".html") or filepath.endswith(".htm"):
            text = self.extract_text_from_html(file_path)
        else:
            with open(file_path, "r") as file:
                text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        base_name = os.path.basename(file_path)
        metadata = [
            {
                "source": base_name,
                "file_path": file_path,
                "chunk_id": i,
                "file_type": os.path.splitext(base_name)[1].lstrip(".").lower()
            }
            for i in range(len(chunks))
        ]
        return chunks, metadata
    
    def process_new_file(self, file_path):
        chunks, metadata = self.load_document(file_path)
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        ids = [f"{meta['source']}_{meta['chunk_id']}" for meta in metadata]
        self.collection.add(documents=chunks, embeddings=embeddings, metadatas=metadata, ids=ids)
        print(f"Processed {file_path} and added to the collection.")

    def list_documents(self):
        all = self.collection.get()
        docs_by_file = {}
        for meta in all['metadatas']:
            source = meta['source']
            docs_by_file[source] = docs_by_file.get(source, 0) + 1
        print("Documents in the collection:")
        for file, count in docs_by_file.items():
            print(f"{file}: {count} chunks")
    
    def delete_document(self, file_basename):
        all_ids = self.collection.get()['ids']
        ids_to_delete = [id for id in all_ids if id.startswith(file_basename)]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks from {file_basename}.")
        else:
            print(f"No chunks found for {file_basename}.")

    def search_memory(self, query, k=5, filetype=None, source_filter=None, min_score=None):
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,
            include=["documents", "metadatas", "distances"]
        )
        filtered_results = []
        for text, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            if filetype and meta['file_type'] != filetype.lower():
                continue
            if source_filter and source_filter not in meta['source']:
                continue
            if min_score and dist > (1- min_score):
                continue
            filtered_results.append((text, meta, dist))

        if not filtered_results:
            print("No results found.")
            return
        
        print(f"\nQuery: {query} \nFound {len(filtered_results)} results:")
        for i, (text, meta, dist) in enumerate(filtered_results[:k]):
            score = 1 - dist
            print(f"[{i}] {meta['source']} (chunk {meta['chunk_id']}, type={meta['file_type']}, score={score:.4f})")
            print(text[:600] + ("..." if len(text) > 600 else ""))
            print("-" * 80)

    
    def watch_folder(self):
        class Handler(FileSystemEventHandler):
            def __init__(self, processor):
                self.processor = processor
            def on_created(self, event):
                print(f"📄 File created: {event.src_path}")
                if not event.is_directory and event.src_path.lower().endswith((".txt", ".pdf", ".docx", ".epub", ".html", ".htm", ".md")):
                    self.processor(event.src_path)

        observer = Observer()
        observer.schedule(Handler(self.process_new_file), self.rag_folder, recursive=False)
        observer.start()
        print(f"Watching {self.rag_folder} for .txt, .pdf, .docx, .epub, .html, .md files... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all stored documents")
    parser.add_argument("--delete", metavar="FILENAME", help="Delete a document by base filename (e.g., 'report.pdf')")
    parser.add_argument("--search", metavar="QUERY", help="Search memory for a query string")
    parser.add_argument("--filter_type", help="Filter by filetype (e.g., 'pdf')")
    parser.add_argument("--filter_source", help="Filter by filename substring")
    parser.add_argument("--min_score", type=float, help="Minimum similarity score (0–1, e.g., 0.85)")
    args = parser.parse_args()

    rag = LocalRAG()

    if args.list:
        rag.list_documents()
    elif args.delete:
        rag.delete_document(args.delete)
    elif args.search:
        rag.search_memory(
            args.search,
            filetype=args.filter_type,
            source_filter=args.filter_source,
            min_score=args.min_score
        )
    else:
        rag.watch_folder()
