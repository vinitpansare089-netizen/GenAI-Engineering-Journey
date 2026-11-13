from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an expert in {field}'),
    ('human', 'Explain me {tech} from real life examples')
])

llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0,
    max_tokens = 512
)

chain = prompt | llm 

result = chain.invoke(
    'field' : 'AI',
    'tech' : 'RAG'
)

print(result.content)


