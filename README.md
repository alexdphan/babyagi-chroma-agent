# Baby AGI template with Langchain Tools and Chroma

Inspired by [Yoheina Kajima's BabyAGI](https://github.com/yoheinakajima/babyagi)

[babyagi-chroma](https://github.com/alexdphan/babyagi-chroma) repository has the advantage of using a free vector storage option (Chroma) locally, which makes it more appealing for users who want to avoid the potential costs associated with other vector storage options such as Pinecone. This difference can be a significant factor when being able to run projects at a cost.

# Objective

This Python script is an example of an AI-powered task management system. The system uses Langchain, OpenAI and Chroma's Vector Database to create, prioritize, and execute tasks. The main idea behind this system is that it creates tasks based on the result of previous tasks and a predefined objective. The script then uses Langchain's OpenAI natural language processing (NLP) capabilities to create new tasks based on the objective, and Chroma to store and retrieve task results for context. This is a pared-down version of the original [Task-Driven Autonomous Agent](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20) (Mar 28, 2023).

This README will cover the following:

- [Baby AGI template with Langchain Tools and Chroma](#baby-agi-template-with-langchain-tools-and-chroma)
- [Objective](#objective)
- [How It Works](#how-it-works)
  - [Execution Agent](#execution-agent)
  - [Task Creation Chain](#task-creation-chain)
  - [Task Prioritization Chain](#task-prioritization-chain)
  - [Chroma](#chroma)
- [How to Use](#how-to-use)
- [Supported Models](#supported-models)
- [Warning](#warning)
- [Contribution](#contribution)
- [Backstory](#backstory)

# How It Works<a name="how-it-works"></a>

The script works by running an infinite loop that does the following steps:

1. Adds and pulls the first task from the task list.
2. Sends the task to the execution agent, which uses Langchain's method, ChatOpenAI(), using OpenAI to complete the task based on the context.
3. Enriches the result and stores it in Chroma.
4. Creates new tasks and reprioritizes the task list based on the objective and the result of the previous task.

## Execution Agent
later
The execution_agent() function is where the OpenAI API is used alongside Langchain. It takes parameters including the objective and the task. It then sends a prompt to Langchain's OpenAI API, which returns the result of the task. The prompt consists of a description of the AI system's task, the objective, and the task itself. The result is then returned as a string. In addition to being an execution agent, the agent is provided with tooling such as search to receive and output more reliable information.

## Task Creation Chain

The class TaskCreationChain is where LLMChain is used to create new tasks. The function from_llm takes in parametersm using PromptTemplate from Langchain, which returns a list of new tasks as strings. The function then creates an instance ofr TaskCreationChain along with the custom input variables and behavior specified.

## Task Prioritization Chain

The class TaskPrioritizationChain is where LLMChain is used to prioritize tasks. The function from_llm takes in parameters using PromptTemplate from Langchain, which returns a list of new tasks as strings. The function then creates an instance of TaskPrioritizationChain along with the custom input variables and behavior specified.


## Chroma

Finally, the script uses Chroma to store, similarity search, and retrieve task results for context. The script creates a Chroma index based on the table name specified in the YOUR_TABLE_NAME variable. Chroma is then used to store the results of the task in the index, along with the task name and any additional metadata.

# How to Use<a name="how-to-use"></a>

To use the script, you will need to follow these steps:

1. Clone the repository via `git clone https://github.com/alexdphan/babyagi-chroma.git` and `cd` into the cloned repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Copy the .env.example file to .env: `cp .env.example .env`. This is where you will set the following variables.
4. Set your OpenAI keys in the OPENAI_API_KEY.
5. ~~There is no Chroma API Key, so we won't have to worry about this.~~
6. Set the name of the table where the task results will be stored in the TABLE_NAME variable.
7. (Optional) Set the objective of the task management system in the OBJECTIVE variable.
8. (Optional) Set the first task of the system in the INITIAL_TASK variable.
9. Run the script using the command `python babyagi-chroma.py`.

All optional values above can also be specified on the command line.

# Supported Models<a name="supported-models"></a>

This script works with all OpenAI models. Default model is **gpt-4**. When there is a timeout request (after two attempts), the model will switch to **gpt-3.5-turbo**. To use a different model, feel free to edit the code however you wish.

# Warning<a name="continous-script-warning"></a>

This script is designed to be run continuously as part of a task management system. Running this script continuously can result in high API usage, so please use it responsibly. Additionally, the script requires the OpenAI API to be set up correctly, so make sure you have set up the APIs before running the script.

# Contribution

To maintain this simplicity, kindly request that you adhere to the following guidelines when submitting PRs:

- Focus on small, modular modifications rather than extensive refactoring.
- When introducing new features, provide a detailed description of the specific use case you are addressing.

# Backstory

With costs of vector storage being expensive, I wanted there to be a free storage option when using BabyAGI. Hence, this is a template example of using BABYAGI with Chroma.

BabyAGI-Chroma is a pared-down version of BabyAGI, of which is also a pared-down version of the original [Task-Driven Autonomous Agent](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20) (Mar 28, 2023) shared on Twitter.
