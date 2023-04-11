from itertools import cycle
import os
from shutil import get_terminal_size
import sys
from threading import Thread
import time
import openai
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from collections import deque
from typing import Dict, List
from langchain.schema import HumanMessage, SystemMessage
import time
import sys
from time import sleep
from rich.console import Console

# Set Variables
load_dotenv()

# Set API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Table / Index / Collection config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Project config
OBJECTIVE = os.getenv("OBJECTIVE", "")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

# This is the first task that the user will be asked to complete
YOUR_FIRST_TASK = os.getenv("FIRST_TASK", "")
assert YOUR_FIRST_TASK, "FIRST_TASK environment variable is missing from .env"

# Print OBJECTIVE
print("\033[96m\033[1m" + "\n***** 🎯 OBJECTIVE 🎯 *****\n" + "\033[0m\033[0m")
print(OBJECTIVE)

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Assigning table_name to YOUR_TABLE_NAME
table_name = YOUR_TABLE_NAME

# Setting embedding to OpenAIEmbeddings, initializing OpenAIEmbeddings with the OpenAI API key
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# openai call: for the executing functions
llm = OpenAI()

# Create the index, specifying the directory where it will be stored
persist_directory = "database"
# Initialize Chroma with the table name, embedding function, and persist directory
vectorstore = Chroma(table_name, embedding, persist_directory=persist_directory)
# Persist the vectorstore to the specified directory
vectorstore.persist()

# Get or create a collection with the specified name and embedding function
collection = vectorstore._client.get_or_create_collection(
    name=table_name, embedding_function=embedding
)

# Initialize an empty task list using the deque data structure
task_list = deque([])


class Loader:
    def __init__(self, desc="Loading...", end="", timeout=0.1):
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        print()  # Add a newline here
        self._thread.start()
        return self

    def _animate(self):
        print()  # Add an empty newline before the animation
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r\033[93m{c} {self.desc} {c}\033[0m", flush=True, end="")
            sleep(self.timeout)
        print()  # Add an empty newline after the animation

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, tb):
        self.stop()


# Define a function to add a task to the task list
def add_task(task: Dict):
    # If the task does not have a task_id value, assign it a default value based on the current length of the task list
    if task["task_id"] is None:
        task["task_id"] = len(task_list)
    # Add the task to the task list
    task_list.append(task)


# Uses model text-embedding-ada-002 by default
def get_ada_embedding(text):
    text = text.replace("\n", " ")
    embeddings = embedding.embed_query(text)
    return [embeddings]  # return a 2D array containing the single embedding


# OpenAI call: for the future executing agent
def openai_call(text, use_gpt3=False, retries=2, retry_wait=15):
    retry_counter = 0
    while retry_counter < retries:
        model_name = "gpt-3.5-turbo" if (use_gpt3 or retry_counter >= 1) else "gpt-4"
        openaichat = ChatOpenAI(
            model_name=model_name,
            max_retries=retries,
            request_timeout=15,
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=text),
        ]
        try:
            response = openaichat(messages)
            return response.content
        except openai.OpenAIError as e:
            if "Request timed out" in str(e):
                print(
                    f"\033[91m"
                    + f"\nTIMEOUT REQUEST USING: {model_name}\n"
                    + "\033[91m"
                )
                print(
                    f"\033[4m"
                    + f"Retrying in {retry_wait} seconds (attempt {retry_counter + 1}/{retries})"
                    + "\033[4m"
                )
                time.sleep(retry_wait)
                retry_counter += 1
                if retry_counter == 1:
                    use_gpt3 = True
                    print("\033[1m" + f"\nUSING {model_name}\n" + "\033[1m")
            else:
                raise e
    print("All retries exhausted. Please try again later.")
    return ""


# Task creation agent
def task_creation_agent(
    task_description: str, result: Dict, task_list: List[str], objective: OBJECTIVE
):
    print("\n")
    print("\033[92m\033[1m" + "\U0001F4DD Task Creation Agent 🎨" + "\033[0m\033[0m")
    create_task = PromptTemplate(
        input_variables=["objective", "result", "task_description", "task_list"],
        template="You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}. The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {task_list}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array.",
    )
    response = openai_call(
        create_task.format(
            objective=objective,
            result=result,
            task_description=task_description,
            task_list=", ".join(task_list),
        )
    )
    new_tasks = response.split("\n")
    return [
        {"task_id": None, "task_name": task_name.strip()}
        for task_name in new_tasks
        if task_name.strip()
    ]


# Prioritization agent
def prioritization_agent(this_task_id: int, gpt_version: str = "gpt-4"):
    global task_list
    task_names = [task["task_name"] for task in task_list]
    next_task_id = int(this_task_id) + 1
    prioritize_task = PromptTemplate(
        input_variables=["task_names", "objective", "next_task_id"],
        template="You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team: {objective}. Do not remove any tasks. Return the result as a numbered list, like:\n#. First task\n#. Second task\nStart the task list with number {next_task_id}",
    )
    response = openai_call(
        prioritize_task.format(
            objective=OBJECTIVE, task_names=task_names, next_task_id=next_task_id
        )
    )
    new_tasks = response.split("\n")
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


# Context agent
def context_agent(query: str, chroma: Chroma, n: int):
    query_embedding = get_ada_embedding(query)
    results = chroma.similarity_search_by_vector(query_embedding[0], k=n)
    return [str(result.page_content) for result in results]


# Execution agent
def execution_agent(objective: str, task: str, chroma: Chroma, n: int):
    context = context_agent(query=task, chroma=vectorstore, n=n)
    execute_task = PromptTemplate(
        input_variables=["objective", "context", "task"],
        template="You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}. Perform the task and return the result.",
    )
    response = openai_call(
        execute_task.format(objective=objective, context=", ".join(context), task=task)
    )
    return response.strip()


# Add the first task to the task list
first_task = {"task_id": 1, "task_name": YOUR_FIRST_TASK}

console = Console()

# Main Loop
try:
    add_task(first_task)
    task_id_counter = 1
    while True:
        if task_list:
            # Display task list
            console.print("\n[bold magenta]*****TASK LIST*****[/bold magenta]")
            for task in task_list:
                console.print(f"[bold]{task['task_id']}. {task['task_name']}[/bold]")

            current_task = task_list.popleft()

            # Display next task
            console.print("\n[bold green]*****NEXT TASK*****[/bold green]")
            console.print(
                f"[bold]{current_task['task_id']}. {current_task['task_name']}[/bold]"
            )

            with Loader("Executing task..."):
                result = execution_agent(
                    objective=OBJECTIVE,
                    task=current_task["task_name"],
                    chroma=vectorstore,
                    n=5,
                )

            # Display task result
            console.print("\n[bold yellow]*****TASK RESULT*****[/bold yellow]")
            console.print(f"[white]{result}[/white]")
            enriched_result = result

            with Loader("Creating new tasks..."):
                new_tasks = task_creation_agent(
                    task_description=current_task["task_name"],
                    result=enriched_result,
                    task_list=[task["task_name"] for task in task_list],
                    objective=OBJECTIVE,
                )

            if new_tasks:  # Check if there are new tasks before printing the message
                console.print("\n[bold blue]*****NEW TASKS CREATED*****[/bold blue]")

            for task in new_tasks:
                console.print(f"[bold]{task['task_id']}. {task['task_name']}[/bold]")
                # Set the ID of each new task to the next available integer value
                task["task_id"] = (
                    str(max([int(t["task_id"]) for t in task_list]) + 1)
                    if task_list
                    else "1"
                )
                add_task(task)

            with Loader("Prioritizing tasks..."):
                prioritization_agent(this_task_id=current_task["task_id"])
            # Sleep before checking the task list again
            time.sleep(1)

except KeyboardInterrupt:
    print("\nScript stopped")
    sys.exit(0)
