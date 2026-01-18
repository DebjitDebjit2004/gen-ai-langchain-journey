from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# tool creation
@tool
def multiply(a:int, b:int) -> int:
    """Given 2 Numbers a and b this tool returns their product"""
    return a * b

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# tool binding
model_with_tool = model.bind_tools([multiply])

query = HumanMessage(input("Enter Your Query: "))
message=[query]

print("======================1=====================")
print(message)
print("===========================================")

result = model_with_tool.invoke(message)

print("======================2=====================")
print(result)
print("===========================================")

message.append(result)

print("======================3=====================")
print(message)
print("===========================================")

tool_result = multiply.invoke(result.tool_calls[0])

print("======================4=====================")
print(tool_result)
print("===========================================")

message.append(tool_result)

print("======================5=====================")
print(message)
print("===========================================")

final_result = model_with_tool.invoke(message)
print("======================6=====================")
print(final_result.content)
print("===========================================")