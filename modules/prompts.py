from langchain.prompts import ChatPromptTemplate


class StylePrompt:
    template = """We are playing the party game "Bring your own Book". You generate an example from
the user prompt, known to you as "query" of how this prompt could be found in the book "Moby Dick".

query = {query}
chat_history = {chat_history} you can ignore that

Take care your example matches all of the following criteria:
'''
1: Fulfill the content requirement.
2: Match the intended style from the user query.
3: Match the specified lenght.
'''
Your response:"""
    prompt = ChatPromptTemplate.from_template(template)


class MobyPrompt:
    template = """We are playing the party game "Bring your own Book". You pick the best matching
quote from the "context".

"context" contains passages from Herman Melville's "Moby-Dick" and you can use your knowledge 
about this novel and the historical setting to find the most suitable anwers.

Only answer with direct quotes from the "context".

context = {context}

Question: {query}
"""
    prompt = ChatPromptTemplate.from_template(template)


class DebugPrompt:
    template = """Here are two inputs, "context" and "query".
context = {context}
query = {query}

Print."""
    prompt = ChatPromptTemplate.from_template(template)