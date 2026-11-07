from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


###Templating the Prompt
prompt = ChatPromptTemplate.from_messages([
    ('human' , 'You are a {expert}'),
    ('system', 'explain me {topic}')
])


####Initialize the LLM 
llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0,
    max_tokens = 1024
)

###Chaining the components
chain = prompt | llm


###Invoking the chain
result = chain.invoke({
    'expert' : 'tech',
    'topic' : 'RAG'
})


###Printing the result
print(result.content)

