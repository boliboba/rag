from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm
from core.config import PROMPTS

THRESHOLD = 0.5

def is_prompt_injection(user_prompt):
    prompt = ChatPromptTemplate.from_template(PROMPTS["security"])
    
    chain = prompt | get_llm() | StrOutputParser()
    
    result = chain.invoke({"prompt": user_prompt})
    
    score = float(result.strip())
    return score > THRESHOLD 