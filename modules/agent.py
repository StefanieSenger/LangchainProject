import os
#from dotenv import load_dotenv; load_dotenv()

from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import Tool

class Agent:
    def __init__(self):
        pass
