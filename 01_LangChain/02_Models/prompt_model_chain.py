from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

### Created a prompt Template

prompt = ChatPromptTemplate.from_template([
    'Exaplin {topic} in simple terms with a real-life example'
])

#### Initialize the LLm 

llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0
)

#### Connects the components

chain = prompt | llm 

##### Now We will Executing the chain

response = chain.invoke({
    'topic' : 'Embeddings'
})

###### Display the results

print(response.content)