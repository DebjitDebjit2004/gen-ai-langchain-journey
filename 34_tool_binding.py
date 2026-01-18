from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

@tool
def multiply(a:int, b:int) -> int:
    """Given 2 Numbers a and b this tool returns their product"""
    return a * b


print(multiply.invoke({'a':3, 'b':4}))

print(multiply.name)
print(multiply.description)