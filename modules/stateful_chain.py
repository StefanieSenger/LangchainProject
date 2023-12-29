from operator import itemgetter
from typing import List, Tuple

from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

from modules.prompts import DebugPrompt, MobyPrompt, StylePrompt
from modules.retriever import Retriever


def instantiate_retriever(docs_path):
    """instantiate retriever with given path to docs"""
    return Retriever(docs_path)

# instantiate retriever
docs_path = "data/id_15.txt"
#docs_path = "data"
retriever = instantiate_retriever(docs_path)

# instantiate prompts
style_prompt = StylePrompt.prompt
retrieval_prompt = DebugPrompt.prompt # <------------------------------------------ DEBUG!


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Formats the chat_history into a string."""
    buffer = ""
    if chat_history:
        human = "Human: " + chat_history[-2].content
        ai = (
            "Assistant: " + chat_history[-1].content
        )
        buffer += "\n" + "\n".join([human, ai])
    return buffer


memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="query"
)

loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
)

standalone_question = {
    "standalone_question": {
        "query": lambda x: x["query"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    }
    | style_prompt
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # model="gpt-4"
    | StrOutputParser()
}

retrieved_context = {
    "context": itemgetter("standalone_question") | retriever.retriever,
    "query": lambda x: x["standalone_question"],
}

final_inputs = {
    "context": itemgetter("context"),
    "query": itemgetter("query"),
}

answer = {
    "answer": final_inputs
    | retrieval_prompt
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser(),
    "metadata": lambda x: [
        document.metadata for document in itemgetter("context")(x)
    ],
}

stateful_chain = loaded_memory | standalone_question | retrieved_context | answer


################## Quick Tests ##################

query = "Find a sentence that would be terrible advice for a DIY project."
result = stateful_chain.invoke({"query": query})
memory.save_context({"query": query}, {"answer": result["answer"]})
print(memory.load_memory_variables({}))

