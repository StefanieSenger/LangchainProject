from dotenv import load_dotenv; load_dotenv()

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import FAISS


class Retriever:
    def __init__(self, query="", docs_path=None):
        self.query = query
        self.docs_path = docs_path
        self.docs_split = None
        self.embedding = None
        self.retriever = None
        self.docs = None
        self.context = ''

    def embed(self):
        # load docs
        loader = TextLoader("../data/id_15.txt") # later: load all docs in directory
        docs = loader.load()

        # split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=100)
        docs_split = text_splitter.split_documents(docs)
        self.docs_split = docs_split

        # make embedding
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002") # try other models
        self.embedding = embedding

        return self

    def retrieve(self):
        # add security check: if not hasattr(obj, self.embedding): raise Error ("bla bla bla") 

        # Initializing FAISS Vector DB not necessary?
        retriever = FAISS.from_documents(self.docs_split, self.embedding).as_retriever(search_kwargs={"k": 3}) # try other retrievers
        self.retriever = retriever

        # run similarity search
        context = retriever.get_relevant_documents(self.query)
        for ele in context:
            self.context += ele.page_content + ele.metadata['source'] + "\n"

        return self