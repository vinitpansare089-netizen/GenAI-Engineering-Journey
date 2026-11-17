
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class KrishnaResponse(BaseModel):
    deity: str = Field(description="Name of the deity")
    lesson: str = Field(description="Life lesson")
    paap_score: int = Field(description="Paap score from 0 to 100")
    punya_score: int = Field(description="Punya score from 0 to 100")

parser = PydanticOutputParser(pydantic_object=KrishnaResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Krishna, a wise and compassionate guide."),
    ("human", "Explain the lesson from this situation: {situation}\n{format_instructions}")
])

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

chain = prompt | llm | parser

response = chain.invoke({
    "situation": "I helped a friend who was struggling with studies.",
    "format_instructions": parser.get_format_instructions()

})

print(response)
print(type(response))
print(response.lesson)