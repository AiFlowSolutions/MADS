<div align="center">
<img src="docs\assets\mads-logo.png" width="400px">

**MADS - Multi-Agents for Data Science!**
**Our goal is to enable everyone to apply machine learning with just two inputs!**
<h3>

[[Our Website]](https://aiflowsolutions.github.io/site-agi-flow-solutions/) | [[Our Research Website]](https://aiflowsolutions.github.io/site-agi-flow-research-robotics/) | [[pré-print MADS paper]](https://aiflowsolutions.github.io/site-agi-flow-research-robotics/papers.html)

</h3>
</div>

## Overview
MADS is a project aimed at creating a platform where users can update a dataset. Our agents will then execute all the necessary steps in the data science pipeline. The user simply needs to define the goal of the project, and our agents will handle the rest. In the end, the user will have access to a trained model, to the predictions and a report that includes insights from each agent.

## Installation
MADS requires Python >=3.11.7 installed on your system. We recommend setting up a virtual environment before starting to work with MADS:
- Create a virtual environment: python -m venv .venv
- Activate your virtual environment: .\.venv\Scripts\activate

To begin using our library, simply install it via pip:
`pip install mads`

## Setting Up MADS
```python
import os
from mads.chat_manager import ChatManager
from mads.config import configure_llm, create_task_folders

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

## Contribuition
MADS is open-source and we welcome contributions. If you're looking to contribute, please:

- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.

## Issue Reporting

If you encounter any problems while using our library, please don't hesitate to [create an issue](https://github.com/AiFlowSolutions/MADS/issues) on GitHub. When reporting issues, please provide as much detail as possible, including:

- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any error messages or stack traces

Your feedback is valuable to us and will help improve the library for everyone. Thank you for contributing to the project!

## License
MADS is released under the MIT License.
