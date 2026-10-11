from dotenv import load_dotenv
from pydantic import BaseModel
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


#LLM set up
llm= ChatOllama(model="nemotron-3-super:cloud ")
llm.invoke("what is the day today").content


class ResearchResponse(BaseModel):
    topic: str
    summary:str
    sources:list[str]
    tools_used: list[str]

parser=PydanticOutputParser(pydantic_object=ResearchResponse)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",

         """ You are a course code, credit, level assistant that will help extract the infomation.
         Answer the user query and use neccessary tools.
         wrap the output in this format and provide no other text \n{format_instructions}"""),

         ("placeholder", "{chat_history}"),
         ("human", "{query}"),
         ("placeholder","{agent_scratchpad}")
    ]
).partail(format_instructions=parser.get_format_instructions())


from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor

agent= create_agent(llm=llm,
                    prompt=prompt,
                    tools=[])


agent_exector=Agent