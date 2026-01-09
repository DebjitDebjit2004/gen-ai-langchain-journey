from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

while True:
    user_query = input('You: ')
    if user_query == 'exit':
        break
    prompt = chat_template.invoke({'chat_history': chat_history, 'query': user_query})
    response = model.invoke(prompt)
    print('AI: ', response.content)

