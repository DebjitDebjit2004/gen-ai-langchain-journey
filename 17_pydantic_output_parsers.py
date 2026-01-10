from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1.0
)

class Person(BaseModel):
    name: str = Field(description='Name of the person', max_length=10)
    age: int = Field(gt=18, lt= 60, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to', max_length=30)
    eligible: bool = Field(description="Is eligible for voting in India")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='name of the person is {name} and age is {age} also place is {place}, is he eligible for voting in india? generate name, age, place and eligibility. \n {format_instruction}',
    input_variables=['name', 'age', 'place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({
    'name': 'Debjit',
    'age': 23,
    'place': 'India'
})

print(final_result)