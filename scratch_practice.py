from langchain_core.prompts import ChatPromptTemplate 

from dotenv import load_dotenv

from langchain_groq import ChatGroq

##### Loading the .env file in program
load_dotenv()

######## Prompt Templating for LLM

prompt = ChatPromptTemplate.from_template(
    'Explain {topic} that if they exists or not'
)

#### connecting the LLM

llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0
)


###### Connecting the components

chain = prompt | llm 


##### Invoking the Prompt only to filling Varible in prompt template
response = chain.invoke({
    'topic': 'GOD'
})

print(response.content)


