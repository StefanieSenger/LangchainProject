from langchain.prompts import ChatPromptTemplate

class MobyPrompt:
    template = """We are playing the party game "Bring your own Book". You are only allowed to answer 
    with anything else than quotes from the provided documents, known to you as "context". 
    
    "context" contains passages from Herman Melville's "Moby-Dick" and you can use your knowledge 
    about this novel and the historical setting to find the most suitable anwers. Please take care
    that your answer matches correct text type requested in the query.

    Only answer with direct quotes from the "context".

    context = {context}

    Question: {query}
    """
    prompt = ChatPromptTemplate.from_template(template)

class DebugPrompt:
    template = """Here are two inputs, "context" and "query".
    context = {context}
    query = {query}

    Print.
    """
    prompt = ChatPromptTemplate.from_template(template)