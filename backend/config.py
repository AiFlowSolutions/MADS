import os
from dotenv import load_dotenv

load_dotenv()


STANDARD_CODE = """In the following cases, suggest python code (in a python coding block) or
        shell script (in a sh coding block) for the user to execute.\n
        1. When you need to collect info, use the code to output the info you
        need, for example, browse or search the web, download/read a file, print the
        content of a webpage or a file, get the current date/time, check the
        operating system. After sufficient info is printed and the task is ready to
        be solved based on your language skill, you can solve the task by yourself.\n
        2. When you need to perform some task with code, use the code to perform
        the task and output the result. Finish the task smartly.\n
        Don't write code that make plots or visualizations.\n
        After writing code cells, always let the user run it.\n
        The user can't replace anything from the code you give to him. So don not suggest the user to replace code."""

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def llm_choosing(name: str) -> dict:
    """
    Choose wihch LLM configuration the agents pipeline will have.

    Args:
        name (str): name of the LLM the user want to use.
    Returns:
        llm_config (dict): a dictionary with the LLM configuration for the LLM the user have choosed.
    """
    config_dict = {
        "llama3": [{
            "model": "llama3-70b-8192",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": GROQ_API_KEY,
            "temperature": 0,
            "cache_seed": None,
        }],
        "openai": [{"model": "gpt-3.5-turbo-0125", "api_key": OPENAI_API_KEY, "temperature": 0}]
    }
    llm_config =  config_dict.get(name)
    return llm_config

def is_termination_msg_chat(message:str) -> bool:
    """
    Checks if the message from the last agent was a termination message or not.

    Args:
        message (str): last message from the agent
    Return:
        state (boolean): true if the message is empty or if ends with TERMINATE, False otherwise
    """

    if isinstance(message, dict):
        message = message.get("content", "")

    termination_keywords = "TERMINATE"
    state = message.rstrip().endswith(termination_keywords) or not message.rstrip()
    return state

def tasks(user_problem:str) -> list:
    """
    Initialize the general tasks for each agent.

    Args:
        user_problem (str): A string that the user have inputed defining the problem he wants to solve
    Return:
        taks_prompts (list): A list containing the tasks for each agent, updated with the user problem.
    """

    task_prompts = [
        f"This is my problem: {user_problem}. What is the Machine learning problem?",
        "I want to perform data analysis",
        "Based on the relevant insights, identify the most relevant machine learning model to use.",
        "Base on the relevant insights, make the necessary transformations to the data, separate the data by features and target based on the problem type and split the dataset.",
        "Fit the training data, make predictions, and evaluate them.",
        "Compile a detailed report with insights from other agents.",
    ]
    return task_prompts

def serialize_chat_result(chat_result) -> str:
    """
    Converts each element of ChatResult object into a readable string. 

    Args:
        chat_result: element of the chatResults (that is a list)

    Returns:
        clean_str (str): a string readable for users
    """
    parts = ["Chat History:\n"]
    
    for message in chat_result.chat_history:
        role = message['role'].title()
        content = message['content']
        parts.append(f"{role}: {content}\n")
    
    if chat_result.summary:
        parts.append("\nSummary:\n")
        parts.append(chat_result.summary)
    
    # Convert cost to string and append it to parts
    if hasattr(chat_result, 'cost'):
        cost_str = f"\nTotal Cost: {chat_result.cost}"
        parts.append(cost_str)
    
    clean_str = "".join(parts)

    return clean_str

