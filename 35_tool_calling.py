from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a:int, b:int) -> int:
    """Given 2 Numbers a and b this tool returns their product"""
    return a * b

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

model_with_tool = model.bind_tools([multiply])

result1 = model_with_tool.invoke('hi, how are you?')
print(result1.tool_calls)

result2 = model_with_tool.invoke('multiply 3 and 10')
print(result2.tool_calls[0])