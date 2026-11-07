from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'explain me {topic}'),
    ('human' , 'You are a {expert}')
])

llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0,
    max_tokens = 1024
)

chain = prompt | llm

result = chain.invoke({
    'expert' : 'tech',
    'topic' : 'RAG'
})

print(result.content)

