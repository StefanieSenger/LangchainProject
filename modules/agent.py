import os

from langchain.chat_models.openai import ChatOpenAI
#from dotenv import load_dotenv; load_dotenv()

from prompts import Baseprompt
from retriever import Retriever, make_data_from_gutenberg

query = "Find a sentence that would be an awkward pick-up line at a bar."

# make data folder
make_data_from_gutenberg(1, 15)

# get additional context from emebedding retriever
retriever = Retriever()
context = retriever.embed().retrieve(query)

# get prompt
pass




# define llm model
llm = ChatOpenAI(model="gpt-3.5", temperature=0)

# define tools
pass

# define langchain agent
pass

# add memory
pass

# define agent executer
pass





# run testsuite (in another file)
pass