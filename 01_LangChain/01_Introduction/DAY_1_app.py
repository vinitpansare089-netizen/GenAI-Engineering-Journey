"""
Scenario: Build a Python Teacher AI.

Requirements:
- System role: Expert Python teacher.
- Human message: Explain Python decorators.
- Ask for two examples.
- Initialize Groq with temperature=0.
- Set max_tokens=1024.
- Connect using LCEL.
- Invoke the chain.
- Print only the generated response.
"""

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq

#### Loading the Environment
load_dotenv()


### Template Prompting
prompt = ChatPromptTemplate.from_messages([

    ('human' , 'You are a Expert {tech} Teacher'),
    ('system' , 'explain {tech} {topic}')
])

### Initialize the LLM
llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 0,
    max_tokens = 512 
)

### chaining the Components
chain = prompt | llm

### invoking and Fills the Template Variables
result = chain.invoke({
    'tech' : 'Python',
    'topic' : 'Decorators'
})

### Print the result 
print(result.content)