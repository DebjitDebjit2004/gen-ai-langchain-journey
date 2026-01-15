from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.5
)

model2 = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.5
)

model3 = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.5
)

loader = DirectoryLoader(
    path='RSP',
    glob='*.pdf',
    loader_cls= PyPDFLoader
)

prompt1 = PromptTemplate(
    template='Give me the key points of the followint text \n {text1}',
    input_variables=['text1']
)

prompt2 = PromptTemplate(
    template='Give me the key points of the followint text \n {text2}',
    input_variables=['text2']
)

prompt3 = PromptTemplate(
    template='Summarize the following key points, key points part 1 -> {kp1} & key points part 2 -> {kp2}'
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'kp1': prompt1 | model1 | parser,
    'kp2': prompt2 | model2 | parser
})

docs = loader.load()

merge_chain = prompt3 | model3 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text1': docs[2].page_content, 'text2': docs[100].page_content})

print(result)

print(chain.get_graph().print_ascii())