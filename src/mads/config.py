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


def configure_llm(llm_name: str, api_key:str) -> dict:
    """
    Choose wihch LLM configuration the agents pipeline will have.

    Args:
        name (str): name of the LLM the user want to use.
    Returns:
        llm_config (dict): a dictionary with the LLM configuraÇtion for the LLM the user have choosed.
    """
    if llm_name == "llama3-70b-8192":
        llm_config=[{
            "model": llm_name,
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": api_key,
            "temperature": 0,
            "cache_seed": None,
            }]
    elif llm_name == "gpt-3.5-turbo-0125":
        llm_config = [{
            "model": llm_name, 
            "api_key": api_key, 
            "temperature": 0,
            "cache_seed": None,
            }]
    else:
        raise ValueError("The specified LLM name doesn't exist in the existing configurations. Please choose another or modify the function in 'config.py'.")
    
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

def create_task_folders():
    """
    Create a folder structure for tasks with datasets and generated files.
    """
    try:
        # Create the main 'tasks' folder
        os.makedirs('tasks', exist_ok=True)
        
        # Create the 'datasets' folder inside 'tasks'
        os.makedirs(os.path.join('tasks', 'datasets'), exist_ok=True)
        
        # Create the 'generated_files' folder inside 'tasks'
        os.makedirs(os.path.join('tasks', 'generated_files'), exist_ok=True)
        
        print("Folders created successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

