
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI teacher. Return the answer in the requested JSON format."),
    ("human", "Give me one life lesson about {topic}.\n{format_instructions}")
])

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

parser = JsonOutputParser()

chain = prompt | llm | parser

response = chain.invoke({
    "topic": "failure",
    "format_instructions": parser.get_format_instructions()
})

print(response)
print(type(response))