from deepeval.models import DeepEvalBaseLLM
from core.llm.models import ChatOpenRouter
import json
import instructor
from openai import OpenAI


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
    
    def generate(self, prompt: str, schema=None) -> str:
        chat_model = self.load_model()
        
        if schema:
            # Используем OpenAI клиент с instructor для структурированного вывода
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.llm.openai_api_key,
            )
            client = instructor.from_openai(client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS)
            
            # Создаем структурированный вывод
            result = client.chat.completions.create(
                model=self.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                extra_body={"provider": {"require_parameters": True}},
            )
            return result
        else:
            # Обычный вызов без схемы
            return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None) -> str:
        chat_model = self.load_model()
        
        if schema:
            # Используем AsyncOpenAI клиент с instructor для структурированного вывода
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.llm.openai_api_key,
            )
            client = instructor.from_openai(client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS)
            
            # Создаем структурированный вывод асинхронно
            result = await client.chat.completions.create(
                model=self.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                extra_body={"provider": {"require_parameters": True}},
            )
            return result
        else:
            # Обычный асинхронный вызов без схемы
            result = await chat_model.ainvoke(prompt)
            return result.content
    
    def get_model_name(self):
        return f"OpenRouter-{self.llm.model_name}" 