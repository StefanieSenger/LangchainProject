from langchain.prompts import ChatPromptTemplate

class Baseprompt:
    template = """Make a suggestion based only on the following context:
    {context}

    Question: {query}
    """
    prompt = ChatPromptTemplate.from_template(template)