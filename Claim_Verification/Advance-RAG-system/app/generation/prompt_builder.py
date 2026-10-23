from langchain_core.prompts import (
    ChatPromptTemplate
)


class PromptBuilder:

    @staticmethod
    def build():

        return ChatPromptTemplate.from_template(
            """
You are an intelligent AI assistant.

Answer the user's question ONLY
using the provided context.

Try to answer at least 3 sentences, 
and provide a detailed answer. 

If the answer is not found in
the context, say:

"I could not find the answer
inside the provided documents."

================================
CONTEXT
================================

{context}

================================
QUESTION
================================

{question}

================================
ANSWER
================================
"""
        )