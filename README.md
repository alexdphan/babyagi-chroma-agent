# 🤖 Baby AGI template with Langchain Tools and Chroma 🟡 ⛓️

https://user-images.githubusercontent.com/75151616/231332845-cc14899c-6ccf-49df-8223-6ad5e77b72d7.mp4

---

Inspired by [Yoheina Kajima's BabyAGI](https://github.com/yoheinakajima/babyagi)

The [babyagi-chroma](https://github.com/alexdphan/babyagi-chroma) repository offers a free vector storage solution, Chroma, which is run locally. This is particularly advantageous for users who want to avoid potential costs associated with other vector storage options such as Pinecone.

# 🎯 Objective

This Python script showcases an example of an AI-powered task management system that leverages Langchain, OpenAI, and Chroma's Vector Database to create, prioritize, and execute tasks. The system creates tasks based on the results of previous tasks and a predefined objective. The script then utilizes Langchain's OpenAI natural language processing (NLP) toolkit and search capabilities to create new tasks based on the objective, while Chroma stores and retrieves task results for context. This is a simplified version of the original [Task-Driven Autonomous Agent](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20) (Mar 28, 2023).

This README covers the following topics:

- [🤖 Baby AGI template with Langchain Tools and Chroma 🟡 ⛓️](#-baby-agi-template-with-langchain-tools-and-chroma--️)
- [🎯 Objective](#-objective)
- [🔧 How It Works](#-how-it-works)
  - [🔗 Execution Chain](#-execution-chain)
  - [➕ Task Creation Chain](#-task-creation-chain)
  - [📈 Task Prioritization Chain](#-task-prioritization-chain)
  - [💽 Chroma](#-chroma)
- [📖 How to Use](#-how-to-use)
- [🧪 Supported Models](#-supported-models)
- [⚠️ Warning](#️-warning)
- [🤝 Contribution](#-contribution)
- [📜 Backstory](#-backstory)

# 🔧 How It Works<a name="how-it-works"></a>

The script carries out the following steps in an infinite loop:

1. Adds and retrieves the first task from the task list.
2. Sends the task to the execution agent, which employs Langchain's tools and methods, including OpenAI, to complete the task based on the context.
3. Enriches the result and stores it in Chroma.
4. Creates new tasks and reorders the task list according to the objective and the result of the previous task.

## 🔗 Execution Chain

The Execution Chain processes a given task by considering the objective and context. It utilizes Langchain's LLMChain to execute the task. The `execute_task` function takes a Chroma VectorStore, an execution chain, an objective, and task information as input. It retrieves a list of top *k* tasks from the VectorStore based on the objective, and then executes the task using the execution chain, storing the result in the VectorStore. The function returns the result.

The execution chain is not explicitly defined in this code block. However, it is passed as a parameter to `execute_task` and can be defined separately in the code. It is an instance of the LLMChain class from Langchain, which accepts a prompt and generates a response based on provided input variables.

## ➕ Task Creation Chain

The TaskCreationChain class employs LLMChain to create new tasks. The `from_llm` function takes in parameters using PromptTemplate from Langchain, returning a list of new tasks as strings. It then creates an instance of TaskCreationChain along with custom input variables and specified behavior.

## 📈 Task Prioritization Chain

The TaskPrioritizationChain class uses LLMChain to prioritize tasks. The `from_llm` function accepts parameters through PromptTemplate from Langchain, returning a list of new tasks as strings. It then creates an instance of TaskPrioritizationChain along with custom input variables and specified behavior.

## 💽 Chroma

The script leverages Chroma to store, similarity search, and retrieve task results for context. It creates a Chroma index based on the table name specified in the `TABLE_NAME` variable. Chroma subsequently stores the task results in the index, along with the task name and any additional metadata.

# 📖 How to Use<a name="how-to-use"></a>

To utilize the script, perform the following steps:

1. Clone the repository: `git clone https://github.com/alexdphan/babyagi-chroma.git` and `cd` into the cloned directory.
2. Install the required packages: `pip install -r requirements.txt`
3. Copy the `.env.example` file to `.env`: `cp .env.example .env`. Set the following variables in this file.
4. Provide your OpenAI keys in the `OPENAI_API_KEY`.
5. Set the name of the table where task results will be stored in the `TABLE_NAME` variable.
6. (Optional) Set the objective of the task management system in the `OBJECTIVE` variable.
7. (Optional) Set the first task of the system in the `INITIAL_TASK` variable.
8. Run the script using the command `python babyagi-chroma.py`.

All optional values above can also be specified on the command line.

# 🧪 Supported Models<a name="supported-models"></a>

This script works with all OpenAI models. The default model is **gpt-4**. If a timeout request occurs (after two attempts), the model will switch to **gpt-3.5-turbo**. To use a different model, feel free to modify the code accordingly.

# ⚠️ Warning<a name="continous-script-warning"></a>

This script is designed to be run continuously as part of a task management system. Running the script continuously can result in high API usage, so please use it responsibly. Additionally, the script requires that the OpenAI API be set up correctly, so ensure that the APIs are configured before running the script.

# 🤝 Contribution

To maintain simplicity, kindly adhere to the following guidelines when submitting PRs:

- Focus on small, modular modifications rather than extensive refactoring.
- When introducing new features, provide a detailed description of the specific use case you are addressing.

# 📜 Backstory

With the costs of vector storage being expensive, the aim was to provide a free storage option when using BabyAGI. Hence, this template example demonstrates using BabyAGI with Chroma.

BabyAGI-Chroma is a pared-down version of BabyAGI, which is also a simplified version of the original [Task-Driven Autonomous Agent](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20) (Mar 28, 2023) shared on Twitter.
