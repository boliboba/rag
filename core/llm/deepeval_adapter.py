from deepeval.models import DeepEvalBaseLLM
from core.llm.models import ChatOpenRouter


class OpenRouterDeepEvalAdapter(DeepEvalBaseLLM):
    
    def __init__(
        self, 
        api_key=None, 
        model_name=None, 
        api_base=None,
        temperature=0.0,
        max_tokens=2000
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
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
    
    def get_model_name(self):
        return f"OpenRouter-{self.llm.model_name}" 