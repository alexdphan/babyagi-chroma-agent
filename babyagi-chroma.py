# Install and Import Required Modules #
import os
from collections import deque
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

from langchain.vectorstores import Chroma

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from chromadb import errors as chromadb_errors

# Set Variables
load_dotenv()

# Setting up and asserting the env var OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Setting up and asserting the env var SERPAPI_API_KEY
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
assert SERPAPI_API_KEY, "SERPAPI_API_KEY environment variable is missing from .env"

# Assigning table_name to YOUR_TABLE_NAME
# Table / Collection config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"
table_name = YOUR_TABLE_NAME

# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# No need to define this as chromadb will use the embedding model to get the embedding size
# embedding_size = 1536
persist_directory = "chromadb"

# Connect to the Vector Store #
# setting vectorstore to Chroma, initializing with table_name, embeddings_model, and persist_directory
vectorstore = Chroma(table_name, embeddings_model, persist_directory=persist_directory)

# Ensuring the vectorstore is persisting to the chromadb
vectorstore.persist()


# Define the Chains #
class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    # BaseLLM takes in a prompt and returns a string
    @classmethod  # new instance of the class every time we call it
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
        # cls is a parameter that allows you to reference the class within the class method. Instead of using the actual class name (TaskCreationChain, in this case)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        # verbose displays additional information, like the prompt and the response
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# Assigning todo_prompt prompt that takes in an objective (prompt) and returns a todo list (response)
todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
# todo_chain is an instance of the LLMChain class
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
# search is an instance of the SerpAPIWrapper class
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
# uses the tools that include search and todo_chain to come up with a todo list
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]

# A prompt that takes in an objective, a task, and a context and returns an answer in zero-shot, outputs in the form of a string
prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)


# Define the BabyAGI Controller #
def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    # response is a string, so we split it into a list of strings
    new_tasks = response.split("\n")
    # return a list of dictionaries, each dictionary has a task_name key value being the task name
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    # task_names is a list of strings, each string is a task name
    task_names = [t["task_name"] for t in task_list]
    # next_task_id is an integer, it is the next task id
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    # prioritized_task_list is a list of dictionaries, each dictionary has a task_id key value being the task id and a task_name key value being the task name
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        # task_parts is a list of strings, the first string is the task id and the second string is the task name
        task_parts = task_string.strip().split(".", 1)
        # if the length of task_parts is 2, then the task id is the first string and the task name is the second string
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    # return the list of dictionaries
    return prioritized_task_list


def _get_top_tasks(vectorstore: Chroma, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    # results is a list of tuples, each tuple has a vectorstore item and a score
    results = vectorstore.similarity_search_with_score(query, k=k)
    # if results is empty, return an empty list
    if not results:
        return []
    # sorted_results is a list of vectorstore items, sorted by score
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    # tasks is a list of strings, each string is a task, for items in sorted_results
    tasks = []
    for item in sorted_results:
        # If the metadata of the item has a task key, then add the task to the list
        try:
            tasks.append(str(item.metadata["task"]))
        # If the metadata of the item does not have a task key, then print a warning
        except KeyError:
            # print(f"Warning: 'task' key not found in metadata of item")
            print(f"")
    return tasks


def execute_task(
    vectorstore: Chroma,
    execution_chain: LLMChain,
    objective: str,
    task_info: Dict[str, Any],
    k: int = 5,
) -> str:
    """Execute a task."""
    k = 5
    # while true, get top k tasks, if not enough, reduce k by 1, if k == 0, break
    while True:
        try:
            context = _get_top_tasks(vectorstore, query=objective, k=k)
            break
        except chromadb_errors.NotEnoughElementsException:
            k -= 1
            if k == 0:
                context = []
                break
    # Execute the task
    result = execution_chain.run(
        objective=objective, context=context, task=task_info["task_name"]
    )
    # store the result on the vectorstore
    result_id = f"result_{task_info['task_id']}"
    vectorstore.add_texts(
        texts=[result],
        metadatas=[
            {"task": task_info["task_name"]}
        ],  # Set 'task' key in metadata here, using task_info
        ids=[result_id],
    )
    return result


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)  # list of tasks
    task_creation_chain: TaskCreationChain = Field(...)  # chain generating new tasks
    task_prioritization_chain: TaskPrioritizationChain = Field(
        ...
    )  # chain prioritizing tasks
    execution_chain: AgentExecutor = Field(...)  # chain executing tasks
    task_id_counter: int = Field(1)  # counter for task ids
    vectorstore: VectorStore = Field(init=False)  # vectorstore for storing results
    max_iterations: Optional[int] = None  # maximum number of iterations

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]

        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )
                # print(f"Stored result {result_id} in Chroma")

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                break
        return {}

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs,
        )


# OBJECTIVE = "Write a weather report for SF today"
OBJECTIVE = "why is babyagi-chroma repo better than the original babyagi repo?"

llm = OpenAI(temperature=0)
# Logging of LLMChains
verbose = False
# If None, will keep on going forever
# We set this to 3 for the sake of the demo
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)
baby_agi({"objective": OBJECTIVE})
