from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.format_messages(
    ('system, You are an expert on {field}'),
    ('human, Explain me {tech} from real life examples')
)

result = prompt.invoke({
    'field' : 'AI',
    'tech' : 'RAG'
})

print(result.content)