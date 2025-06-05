import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Не найден OPENAI_API_KEY! Проверь .env файл")

from llama_index.core import ( # Changed import
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings, # Use Settings instead of ServiceContext
)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI # Use LlamaIndex's OpenAI LLM
# from langchain.llms import OpenAI # No longer directly needed if using LlamaIndex's OpenAI LLM

class LlamaIndexAgent:
    def __init__(self, rebuild_index=True):
        # Configure the LLM directly through LlamaIndex's Settings
        # No need for LangChainLLMPredictor in this setup
        Settings.llm = LlamaIndexOpenAI(api_key=OPENAI_API_KEY)
        Settings.chunk_size = 1024 # Example setting, adjust as needed
        Settings.chunk_overlap = 20 # Example setting, adjust as needed

        self.index_dir = "indexes"
        self.docs_dir = "data/people_docs"

        if rebuild_index or not os.path.exists(self.index_dir):
            print("[LlamaIndexAgent] Анализирую людей...")
            documents = SimpleDirectoryReader(self.docs_dir).load_data()
            self.index = VectorStoreIndex.from_documents(documents) # Settings are global or passed implicitly
            self.index.storage_context.persist(persist_dir=self.index_dir)
        else:
            print("[LlamaIndexAgent] Загружаю инфо о людях...")
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            self.index = load_index_from_storage(storage_context) # Settings are global or passed implicitly

        self.query_engine = self.index.as_query_engine()

    def query_knowledge(self, question: str) -> str:
        print(f"[LlamaIndexAgent] Вопрос: {question}")
        response = self.query_engine.query(question)
        return str(response)

# Example usage (for testing)
if __name__ == "__main__":
    agent = LlamaIndexAgent(rebuild_index=True) # Set to True to rebuild the index initially
    response = agent.query_knowledge("What is LlamaIndex?")
    print(f"[LlamaIndexAgent] Ответ: {response}")