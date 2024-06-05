<div align="center">
<img src="docs\assets\mads-logo.png" width="400px">

**MADS - Multi-Agents for Data Science!**
**Our goal is to enable everyone to apply machine learning with just two inputs!**

[[Our Website]](https://www.aiflowsolutions.pt/)
[[pré-print MADS paper]](docs\assets\MADS__Multi-agents-for-data-science.pdf)
[[Docs]]()

</div>

## Overview
MADS is a project aimed at creating a platform where users can update a dataset. Our agents will then execute all the necessary steps in the data science pipeline. The user simply needs to define the goal of the project, and our agents will handle the rest. In the end, the user will have access to a trained model, to the predictions and a report that includes insights from each agent.


## Setting Up MADS
```python
import os
from src.mads.chat_manager import ChatManager
from src.mads.config import configure_llm, create_task_folders

# Check if the 'tasks' directory exists, if not, create it.
if not os.path.exists('tasks'):
    create_task_folders()

# After creating the folder, upload the dataset (a csv) you wish to test into tasks/datasets. 
# Note: If there is no dataset in tasks, the pipeline will proceed but yield no results.

# Set your API keys here.
# These keys may be from Groq or OpenAI; current configurations are better suited for Groq.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Select the model to use (tested options include 'llama3-70b-8192' and 'gpt-3.5-turbo-0125').
# To test additional models, modify the config.py script accordingly.
model = configure_llm("llama3-70b-8192", GROQ_API_KEY)

# Define the supervised ML problem that you intend to solve.
problem = "I want to predict wine quality"

# Specify the filename of your data
dataset = "winequality-red"

# Choose the agents to deploy.
# Here, we select all six available agents.
agents = [1, 2, 3, 4, 5, 6]

# Configure the ChatManager class.
chat_manager = ChatManager(dataset, problem, model, agents)

# Begin the chat sessions.
chat_results = chat_manager.initiate_chats()
```

