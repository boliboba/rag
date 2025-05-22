from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.llm import get_llm
from core.config import PROMPTS, DOCS_SEPARATOR
from core.db import retrieve, rerank


def format_docs(docs):
    contents = []
    for doc in docs:
        # Проверяем, является ли элемент объектом документа с атрибутом page_content
        if hasattr(doc, 'page_content'):
            contents.append(doc.page_content)
        else:
            # Если это строка или другой тип, преобразуем в строку
            contents.append(str(doc))
    
    return DOCS_SEPARATOR.join(contents)


def split_docs(context):
    """Разделяет контекст на отдельные документы по разделителю"""
    return context.split(DOCS_SEPARATOR)


def get_qa_chain():
    prompt = PromptTemplate.from_template(PROMPTS["qa"])
    llm = get_llm()
    return prompt | llm | StrOutputParser()


def get_retrieval_chain(query_transformer=None):
    prompt = ChatPromptTemplate.from_template(PROMPTS["qa"])
    
    llm = get_llm()
    
    def retrieve_and_rerank(query):
        docs = retrieve(query, query_transformer=query_transformer)
        return rerank(query, docs)
    
    retrieval_fn = lambda query: format_docs(
        retrieve_and_rerank(query)
    )
    
    rag_chain = (
        {"context": retrieval_fn, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain 