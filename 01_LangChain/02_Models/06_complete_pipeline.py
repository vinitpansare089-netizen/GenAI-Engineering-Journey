"""
Write the code yourself with these requirements:
1. Create a prompt with two variables: {deity} and {topic}.
2. Set the system message to define a wise, compassionate guide.
3. Initialize ChatGroq with temperature 0 and max_tokens 1024.
4. Initialize StrOutputParser().
5. Connect all three components using |.
6. Invoke the chain with:
   - deity: Krishna
   - topic: How should we handle failure in life?
7. Print the answer and its Python type.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are a {deity}'),
    ('human', 'give me lessons on {topic}')
])

llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=1,
    max_tokens=400
)

parser = JsonOutputParser()

chain = prompt | llm | parser

response = chain.invoke({
    'deity' : 'Krishna',
    'topic' : 'Life'
})

# print("Response:", response)
print("Type:", type(response))
print("Is string:", isinstance(response, str))
print("Length:", len(response))