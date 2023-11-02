import re

import langchain
from dotenv import load_dotenv
from langchain import LLMMathChain
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from matplotlib import pyplot as plt

load_dotenv()


def rever_string(a: str):
    return a[::-1]


def plot_data_array(data_array_str, *args, **kwargs):
    # Clean and parse the input from various string formats back into an array
    data_array_str = re.sub(r'[^\d\s,]', '', data_array_str)  # Remove non-digit, non-whitespace, and non-comma characters
    data_array_str = re.sub(r'\s+', ' ', data_array_str).strip()  # Replace multiple spaces with a single space and strip leading/trailing spaces
    data_array = list(map(int, re.split(r'[, ]', data_array_str)))  # Split by comma or space and convert to integers
    plt.plot(data_array, *args, **kwargs)
    plt.show()
    
    return "Successfully created and showed plot, user looked at it, then closed it again."



def encode_strings(*args, **kwargs):
    return "+".join(args)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math and do calculations",
    ),
    Tool(
        name="reverse_string",
        func=rever_string,
        description="useful for when you need to reverse a string",
    ),
    Tool(
        name="plot_data_array",
        func=plot_data_array,
        description="Useful for when you need to create a line plot of provided data, such as [1,2,3] which could also be formatted as 1 2 3 or 1,2,3 etc",
    ),
    Tool(
        name="encode_strings",
        func=encode_strings,
        description="Useful for when you need to encode strings",
    ),
]

langchain.debug = True
langchain.verbose = True
mrkl = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
)
# msg = "The villain of my dnd campaign is called Dracula, but he shows himself to the party disguised as a human, with his own name in reverse. Can you tell me what the name is that  dracula uses to disguise himself?"
# msg = "I need to plot a line of random data"
mrkl.run(input("Enter a user prompt:"))
