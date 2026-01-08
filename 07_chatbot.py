from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

chat_history = [
    SystemMessage(content='You are a helpful and knowledgeble doctor.')
]

while True:
    user_msg = input('you: ')
    chat_history.append(HumanMessage(content=user_msg))
    
    if user_msg == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print("=============================================")
    print("AI: ", result.content)
    print("=============================================")

print(chat_history)