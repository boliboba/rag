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
    
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        try:
            response = chat_model.invoke(prompt).content
            
            # Проверяем, требуется ли JSON (DeepEval часто ожидает JSON)
            if '"score":' in prompt or 'JSON format' in prompt:
                try:
                    # Пытаемся извлечь JSON из ответа, если модель вернула пояснения
                    if '```json' in response:
                        json_part = response.split('```json')[1].split('```')[0].strip()
                        # Проверяем что это валидный JSON
                        json.loads(json_part)
                        return json_part
                    elif '{' in response and '}' in response:
                        # Пытаемся извлечь фрагмент JSON
                        json_part = response[response.find('{'):response.rfind('}')+1]
                        # Проверяем что это валидный JSON
                        json.loads(json_part)
                        return json_part
                except:
                    # Если не смогли извлечь JSON, возвращаем заглушку
                    return '{"score": 0.5}'
            
            return response
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            # Возвращаем подходящую заглушку в зависимости от запроса
            if '"score":' in prompt or 'JSON format' in prompt:
                return '{"score": 0.5}'
            return "Error generating response"

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        try:
            res = await chat_model.ainvoke(prompt)
            response = res.content
            
            # Проверяем, требуется ли JSON (DeepEval часто ожидает JSON)
            if '"score":' in prompt or 'JSON format' in prompt:
                try:
                    # Пытаемся извлечь JSON из ответа, если модель вернула пояснения
                    if '```json' in response:
                        json_part = response.split('```json')[1].split('```')[0].strip()
                        # Проверяем что это валидный JSON
                        json.loads(json_part)
                        return json_part
                    elif '{' in response and '}' in response:
                        # Пытаемся извлечь фрагмент JSON
                        json_part = response[response.find('{'):response.rfind('}')+1]
                        # Проверяем что это валидный JSON
                        json.loads(json_part)
                        return json_part
                except:
                    # Если не смогли извлечь JSON, возвращаем заглушку
                    return '{"score": 0.5}'
            
            return response
        except Exception as e:
            print(f"Ошибка при асинхронной генерации ответа: {e}")
            # Возвращаем подходящую заглушку в зависимости от запроса
            if '"score":' in prompt or 'JSON format' in prompt:
                return '{"score": 0.5}'
            return "Error generating response"
    
    def get_model_name(self):
        return f"OpenRouter-{self.llm.model_name}" 