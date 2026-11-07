from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    'you are {tech} an expert make me understand with an example'
)

result = 