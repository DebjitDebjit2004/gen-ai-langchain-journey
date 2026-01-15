from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.5
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/samsung-galaxy-s24-ultra-5g-titanium-violet-256-gb/p/itm40a42309a03b1?pid=MOBH3P4UHZHQVRYT&lid=LSTMOBH3P4UHZHQVRYTQWBAXK&marketplace=FLIPKART&q=samsung+s24+ultra&store=tyy%2F4io&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_5_na_na_ps&otracker1=AS_QueryStore_OrganicAutoSuggest_1_5_na_na_ps&fm=organic&iid=279f9439-9e6f-4c04-9616-6d920d9858e3.MOBH3P4UHZHQVRYT.SEARCH&ppt=hp&ppn=homepage&ssid=pchcer0bio0000001768470563663&qH=be291326d56ab781'

loader = WebBaseLoader(url)

docs = loader.load()

print("====================================================================")
print(docs)
print("====================================================================")

chain = prompt | model | parser

print("====================================================================")
print(chain.invoke({'question':'What is the price of this product?', 'text':docs[0].page_content}))
print("====================================================================")
