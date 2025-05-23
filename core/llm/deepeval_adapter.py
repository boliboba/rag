from deepeval.models import DeepEvalBaseLLM
import instructor
from openai import OpenAI, AsyncOpenAI
import os


class OpenRouterDeepEvalAdapter(DeepEvalBaseLLM):
    
    def __init__(
        self, 
        api_key=None, 
        model_name=None, 
        api_base=None,
        temperature=0.0,
        max_tokens=10000
    ):
        # Получаем API ключ из переменной окружения, если не передан
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base or "https://openrouter.ai/api/v1"
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def load_model(self):
        # Создаем и возвращаем OpenAI клиент с instructor
        client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )
        return instructor.from_openai(client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS)
    
    def generate(self, prompt: str, schema=None) -> str:
        client = self.load_model()
        
        if schema:
            # Создаем структурированный вывод
            result = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"provider": {"require_parameters": True}},
            )
            return result
        else:
            # Обычный вызов без схемы
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        # Создаем асинхронный клиент
        async_client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )
        async_client = instructor.from_openai(async_client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS)
        
        if schema:
            # Создаем структурированный вывод асинхронно
            result = await async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"provider": {"require_parameters": True}},
            )
            return result
        else:
            # Обычный асинхронный вызов без схемы
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
    
    def get_model_name(self):
        return f"OpenRouter-{self.model_name}" 