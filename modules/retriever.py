from dotenv import load_dotenv; load_dotenv()

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import FAISS



class Retriever:
    def __init__(self, docs_path=None, k=15, lambda_mult=1):
        self.docs_path = docs_path
        self.k = k
        self.lambda_mult = lambda_mult
        self.docs = None
        self.docs_split = None
        self.embedding = None
        self.retriever = None
        self.context = ''

        self._retriever_init()

        return None        

    def _retriever_init(self):
        # make embedding
        openai_embedding = OpenAIEmbeddings(
            model="text-embedding-ada-002", chunk_size=1500
        )

        self.embedding = openai_embedding

        # load docs
        loader = TextLoader(self.docs_path) # later: load all docs in directory
        self.docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=20)
        docs_split = text_splitter.split_documents(self.docs)
        self.docs_split = docs_split

        faiss_data = FAISS.from_documents(self.docs_split, self.embedding)

        # use FAISS data as retriever
        retriever = faiss_data.as_retriever(
            search_type="mmr",  # "similarity", "mmr", "similarity_score_threshold"
            search_kwargs={
                # "score_threshold": 0.5,
                "k": self.k,  # number of retrieved documents
                "lambda_mult": self.lambda_mult,  # value between 0 and 1, controls for dissimilarity of retrievals
                "include_metadata": True,
            },
        )
        self.retriever = retriever

        


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