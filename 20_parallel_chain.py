from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

model3 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template='Rectify the gramatically problems as a point of the following code \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Rectify the syntax problems as a point of the following code \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided gramatically Problem and Syntax Problem into a single document \n gramatically problems -> {gp} and syntax problems -> {sp} \n also write the correct code.',
    input_variables=['gp', 'sp']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'gp': prompt1 | model1 | parser,
    'sp': prompt2 | model2 | parser
})

text = '''
def calculateSum(a b)
    print("This function calculate sum of two number")
    result = a + b
    if result > 10
        print("Result is greater then ten")
    else
        print("Result are small")
    return result

x = 5
y = "3"

total = calculateSum(x, y
print("Total is :", total)
'''

merge_chain = prompt3 | model3 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text': text})

print(result)

print(chain.get_graph().print_ascii())