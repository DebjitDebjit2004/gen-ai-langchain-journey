from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

messages = [
    SystemMessage(content='You are a knowledgeble doctor'),
    HumanMessage(content='Tell me about some acidity relief teablets name')
]

result = model.invoke(messages)

ai_msg = AIMessage(content=result.content)
messages.append(AIMessage(content=result.content))
print(messages)