from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are {diety}'),
    ('human', 'give me views on {topic}')
])

llm = ChatGroq(
    model = 'openai/gpt-oss-20b',
    temperature = 1,
    max_tokens = 300
)

####Parsing the output in string
parser = StrOutputParser()

###Chaining the components
chain = prompt | llm | parser



