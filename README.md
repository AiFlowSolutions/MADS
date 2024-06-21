<div align="center">
<img src="docs\assets\mads-logo.png" width="400px">

**MADS - Multi-Agents for Data Science!**
**Our goal is to enable everyone to apply machine learning with just two inputs!**
<h3>

[[Our Website]](https://aiflowsolutions.github.io/site-agi-flow-solutions/) | [[Our Research Website]](https://aiflowsolutions.github.io/site-agi-flow-research-robotics/) | [[pré-print MADS paper]](https://aiflowsolutions.github.io/site-agi-flow-research-robotics/papers.html)

[![PyPI](https://img.shields.io/pypi/v/pymads)](https://pypi.org/project/pymads/)

</h3>
</div>

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Setting up MADS](#setting-up-mads)
- [Contribuition](#contribuition)
- [Issue Reporting](#issue-reporting)
- [Contact us](#contact-us)
- [License](#license)


## Overview
MADS is a project aimed at creating a platform where users can update a dataset. Our agents will then execute all the necessary steps in the data science pipeline. The user simply needs to define the goal of the project, and our agents will handle the rest. In the end, the user will have access to a trained model, to the predictions and a report that includes insights from each agent.

## Roadmap
- Implement a Reiforcment Learning Agent to improve the current prompts.
- Create tools for the agents to use them.
    - Example 1: For the model-building agent, provide a tool like one from Nixtla's library to improve forecasting for time series problems.
    - Example 2: For the model consultant agent, create a tool to optimize the selected model.
- Introduce a new agent to interact with the data analyst and generate visualizations useful for the final report.

## Installation
MADS requires Python >=3.11.7 installed on your system. We recommend setting up a virtual environment before starting to work with MADS:
- Create a virtual environment: `python -m venv .venv`
- Activate your virtual environment: `.\.venv\Scripts\activate`

To begin using our library, simply install it via pip:
`pip install pymads`

## Setting Up MADS
Before running the bellow script create a `.env` file and place your API Key inside it: `GROQ_API_KEY="your_api_key"` or `OPENAI_API_KEY="your_api_key"`

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

## Contact Us

If you have any questions, suggestions, or feedback regarding MADS, please feel free to reach out to us:

- **Company Email**: [info@aiflowsolutions.eu](mailto:info@aiflowsolutions.eu)

##### Main Contributors

- **Email**: [wutsuperwut@gmail.com](mailto:wutsuperwut@gmail.com)
- **Email**: [diogofranciscop@hotmail.com](mailto:diogofranciscop@hotmail.com)

We are committed to improving MADS based on your input and look forward to hearing from you!

## Ackonwledgments
A heartfelt thank you to all the contributors of the autogen framework. Your dedication and hard work have been instrumental in making this project possible. We deeply appreciate the entire community's support and involvement.

## License
MADS is released under the MIT License.
