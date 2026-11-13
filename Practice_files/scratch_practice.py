from langchain_core.prompts import ChatPromptTemplate 

from dotenv import load_dotenv

from langchain_groq import ChatGroq

##### Loading the .env file in program
load_dotenv()

######## Prompt Templating for LLM

prompt = ChatPromptTemplate.from_messages([
    ('system' , 'you ara a {role}'),
    ('ai' , 'Do not give diplomatic response on {topic}')
])

#### connecting the LLM

llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0,
    max_tokens = 1024
)

###### Connecting the components

chain = prompt | llm 


##### Invoking the Prompt only to filling Varible in prompt template
response = chain.invoke({
    'role' : 'Atheist',
    'topic': 'GOD'
})


print(response.content)

