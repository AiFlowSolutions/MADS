# MADS - Experiences for our paper 
[[pré-print MADS paper]](assets\MADS_Multi_Agents-For_Data_Science.pdf)

## Overview
In this branch, you will find the exact experiments that we have reproduced in our paper. Feel free to replicate the results; however, please note that due to the inherent randomness of large language models (LLMs), the results may not be identical.

## Replicate us guide
- Ensure you have python installed on your system.
- Clone our repository locally by executing the following command in your terminal: `git clone https://github.com/AiFlowSolutions/MADS`
- Navigate to the folder MADS.
- Switch from the main branch to this branch:: `git checkout experimental-results-paper`
- Create a virtual enviroment: `python -m venv .venv`
- Activate your virtual enviroment: `.\.venv\Scripts\activate`
- Install the required packages: `pip install -r requirements.txt`
- Create a .env file and place your API Key inside it: `GROQ_API_KEY="your_api_key"`
- Run the experiment: `python dev_testing.py`

**NOTE**: We advise you to clean the `chat_history` folder and the `generated_files` folder to ensure that the files generated are from your experiment.

## Results analysis
If you would like to check our results and conduct your own analysis, examine the files in the chat_history folder, where you will find the chat history for each dataset experiment. The generated files from the experiments are in the tasks/generated_files folder. Note that the models in the pickle format are not included here due to their large size, which exceeded GitHub's storage limits even with LFS.

If you find any discrepancies between the paper and the results, or if you would like to provide feedback, please feel free to contact us by sending an email to `wutsuperwut@gmail.com` or `info@aiflowsolutions.eu`.
