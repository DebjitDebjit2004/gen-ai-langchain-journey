from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

loader = TextLoader('cricket.txt', encoding='utf8')
parser = StrOutputParser()

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))