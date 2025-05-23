from deepeval.models import DeepEvalBaseLLM
from core.llm.models import ChatOpenRouter
import json


class OpenRouterDeepEvalAdapter(DeepEvalBaseLLM):
    
    def __init__(
        self, 
        api_key=None, 
        model_name=None, 
        api_base=None,
        temperature=0.0,
        max_tokens=10000
    ):
        # Создаем экземпляр LangChain модели
        self.llm = ChatOpenRouter(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    def load_model(self):
        return self.llm
    
    def generate(self, prompt: str, schema) -> str:
        chat_model = self.load_model()
        result = chat_model.with_structured_output(schema).invoke(prompt).content
        json_result = json.loads(result)
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema) -> str:
        chat_model = self.load_model()
        result = await chat_model.with_structured_output(schema).ainvoke(prompt)
        json_result = json.loads(result.content)
        return schema(**json_result)
    
    def get_model_name(self):
        return f"OpenRouter-{self.llm.model_name}" 