import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env file
load_dotenv()

# Initialize Groq LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=500
)

# Invoke model
response = model.invoke(
    "tell me about indian cricket. answer with one line"
)

print(response.content)
