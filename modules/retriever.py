import os
import requests
from dotenv import load_dotenv; load_dotenv()

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import FAISS


def make_data_from_gutenberg(start, stop):
    """creates a directory consisting of txt files"""
    if not os.listdir("data"):
        for book_id in range(start, stop+1):
            url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'

            try:
                response = requests.get(url)
                book_text = response.text
                file_path = f"data/id_{book_id}.txt"

                with open(file_path, 'w') as file:
                    file.write(book_text)
            except:
                pass


class Retriever:
    def __init__(self):
        pass

    def embed(self):
        # load docs
        loader = TextLoader("data/id_15.txt") # later: load all docs in directory
        docs = loader.load()

        # split text
        text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=100,) # some chunks are significantly larger than 1000, some up to over 2000
        docs_split = text_splitter.split_documents(docs)
        self.docs_split = docs_split

        # make embedding
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002") # try other models
        self.embedding = embedding

        return self

    def retrieve(self, query):
        # add security check: if not hasattr(obj, self.embedding): raise NotImplementedError(?) ("bla bla bla") 

        # Initializing FAISS Vector DB not necessary?
        retriever = FAISS.from_documents(self.docs_split, self.embedding).as_retriever(search_kwargs={"k": 3}) # try other retrievers

        # run similarity search
        context = retriever.get_relevant_documents(query)

        return context