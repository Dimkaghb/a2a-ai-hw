# agents/agent_langchain.py

from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from agents.agent_llamaindex import LlamaIndexAgent


class LangchainAgent:
    def __init__(self):
        # Создаём экземпляр агента LlamaIndex
        self.llama_agent = LlamaIndexAgent()

        # Создаём Langchain-инструмент, который вызывает query_knowledge
        tools = [
            Tool(
                name="LlamaIndex Knowledge Search",
                func=self.llama_agent.query_knowledge,
                description="Полезен, когда нужно найти информацию в документах",
            )
        ]

        # Создаём Langchain агента, который умеет использовать инструменты
        self.agent = initialize_agent(
            tools=tools,
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self, question: str) -> str:
        return self.agent.run(question)
