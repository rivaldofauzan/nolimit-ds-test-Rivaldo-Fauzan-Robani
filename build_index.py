import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = PyPDFDirectoryLoader("./data/")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(docs)} chunks.")


model_name = "paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(model_name)


texts = [doc.page_content for doc in docs]
embeddings = embedding_model.encode(texts, show_progress_bar=True)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))


faiss.write_index(index, "./vector_store/docs.index")

with open("./vector_store/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Indexing complete and saved to ./vector_store/")