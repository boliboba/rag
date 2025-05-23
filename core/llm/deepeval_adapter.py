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
    
    def generate(self, prompt: str, schema) -> str:
        try:
            client = self.load_model()
            result = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=schema,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"provider": {"require_parameters": True}},
                )
            return result
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            return None

    async def a_generate(self, prompt: str, schema) -> str:
        try:
            async_client = AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
            async_client = instructor.from_openai(async_client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS)

            result = await async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"provider": {"require_parameters": True}},
                )
            return result
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            return None
    
    def get_model_name(self):
        return f"OpenRouter-{self.model_name}" 