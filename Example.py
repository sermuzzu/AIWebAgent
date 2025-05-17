# ai_mcp_agent_example.py
# Comprehensive example of an AI agent using the MCP protocol in Python,
# plus demonstrations of core Python concepts with thorough comments
# to aid understanding for beginners.

import os                        # Provides functions for interacting with the operating system
import json                      # Allows reading from and writing to JSON files
import time                      # Provides time-related functions (e.g., sleep)
import random                    # Offers random number generation and choices
import math                      # Supplies mathematical functions (e.g., sqrt)
from typing import List, Dict, Any, Iterator  # Used for type hints to clarify variable types
import functools                 # Contains tools for working with functions (e.g., decorators)
import openai                   # Azure OpenAI client (install via `pip install openai`)
from sklearn.metrics.pairwise import cosine_similarity   # To measure similarity between vectors
import numpy as np                                      # For array operations required by sklearn
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv  # NEW: for loading .env

# Load environment variables from .env file
load_dotenv()

# ------------------------
# 1. Configuration & Constants
# ------------------------
# Azure OpenAI configuration
# For OpenAI Python SDK >=1.0.0, use AzureOpenAI client instead of setting api_base/api_type directly
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')  # Use the version your Azure resource supports

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_API_VERSION:
    raise RuntimeError("Azure OpenAI credentials are not set. Please check your .env file.")

client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# File path where the agent's memory entries are stored
MEMORY_FILE = 'agent_memory.json'
# Azure OpenAI deployment names (not model names!)
EMBEDDING_DEPLOYMENT = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
CHAT_DEPLOYMENT = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-4o-mini')

# ------------------------
# 2. Helper Functions
# ------------------------

def load_memory() -> List[Dict[str, Any]]:
    """
    Load saved memory entries from a JSON file.
    Each entry is a dict with 'text' (the memory) and 'embedding' (vector).
    If the file doesn't exist, return an empty list.
    """
    if not os.path.exists(MEMORY_FILE):
        return []  # No memories saved yet
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)


def save_memory(memories: List[Dict[str, Any]]) -> None:
    """
    Save the list of memory entries to a JSON file,
    with pretty indentation for readability.
    """
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memories, f, indent=2)

def embed_text(text: str) -> list:
    """
    Create an embedding for the given text using Azure OpenAI.
    """
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

# ------------------------
# 3. Memory Component
# ------------------------
class Memory:
    """
    Handles storing new entries and retrieving similar past entries
    using vector embeddings and cosine similarity.
    """
    def __init__(self):
        # Load existing memories on initialization
        self.memories = load_memory()

    def add(self, text: str):
        # Create embedding for the new memory text
        emb = embed_text(text)
        # Append the new entry to the memory list
        self.memories.append({'text': text, 'embedding': emb})
        # Save updated memories back to disk
        save_memory(self.memories)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        # If no memories yet, return empty list
        if not self.memories:
            return []
        # Embed the query to compare against saved embeddings
        query_emb = embed_text(query)
        # Extract all saved embeddings into a list
        embeddings = [entry['embedding'] for entry in self.memories]
        # Convert lists to numpy arrays for sklearn compatibility
        query_emb_np = np.array([query_emb])
        embeddings_np = np.array(embeddings)
        # Compute cosine similarity scores between query and each memory
        similarities = cosine_similarity(query_emb_np, embeddings_np)[0]
        # Pair each score with its memory entry and sort descending
        ranked = sorted(zip(similarities, self.memories), key=lambda x: -x[0])
        ranked = sorted(zip(similarities, self.memories), key=lambda x: -x[0])
        # Return the top_k memory texts only
        return [entry['text'] for _, entry in ranked[:top_k]]

# ------------------------
# 4. Context Management
# ------------------------
class ContextManager:
    """
    Builds a string combining the goal, last action, and relevant memories
    to provide context for planning.
    """
    def __init__(self, memory: Memory):
        self.memory = memory

    def build(self, goal: str, last_action: str = '') -> str:
        parts = []
        # Include the high-level goal
        parts.append(f"Goal: {goal}")
        # Optionally include the last action executed
        if last_action:
            parts.append(f"Last action: {last_action}")
        # Retrieve similar past memories to the goal
        relevant_memories = self.memory.retrieve(goal)
        if relevant_memories:
            parts.append("Relevant memories:")
            # List each memory on its own line, prefixed by '- '
            parts.extend(f"- {mem}" for mem in relevant_memories)
        # Join all parts with line breaks for the prompt
        return "\n".join(parts)

# ------------------------
# 5. Planning with LLM
# ------------------------
class Planner:
    """
    Sends the context to an LLM (e.g., GPT) and receives a concise action command in return.
    """
    def __init__(self, deployment: str = CHAT_DEPLOYMENT):
        self.deployment = deployment

    def plan(self, context: str) -> str:
        # Add a system message to enforce safe, professional output
        system_message = "You are a helpful, safe, and professional AI agent. Only suggest actions that are safe for work and appropriate for all audiences."
        prompt = (
            "You are an AI agent. Given the context, decide the next action.\n\n"
            + context + "\nNext action:"
        )
        response = client.chat.completions.create(
            model=self.deployment,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content
        return content.strip() if content is not None else ""

# ------------------------
# 6. Tool Functions
# ------------------------
def web_search(query: str) -> str:
    """
    Perform a Google search and return the top 3 result titles and links.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    params = {"q": query}
    url = "https://www.google.com/search"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for g in soup.select('div.g'):
            title = g.select_one('h3')
            link = g.select_one('a')
            if title and link:
                results.append(f"{title.text.strip()}\n{link['href']}")
            if len(results) >= 3:
                break
        if results:
            return "Top Google results:\n" + "\n\n".join(results)
        else:
            return "No results found."
    except Exception as e:
        return f"Web search error: {e}"


def calculator(expr: str) -> str:
    """
    Safely evaluate a simple math expression provided as a string.
    Uses restricted eval environment to prevent code injection.
    Returns the result or an error message.
    """
    try:
        # '__builtins__' disabled to prevent unauthorized code execution
        result = eval(expr, {'__builtins__': {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"

# ------------------------
# 7. Agent Orchestration (MCP Loop)
# ------------------------
class Agent:
    """
    Implements the Memory → Context → Planning → Action → Memory cycle.
    """
    def __init__(self, goal: str):
        self.goal = goal
        self.memory = Memory()
        self.context_mgr = ContextManager(self.memory)
        self.planner = Planner()
        self.last_action = ''

    def perceive(self) -> str:
        # For this example, perception is simply tracking the last action
        return self.last_action

    def act(self, action: str) -> str:
        if action.startswith('search:'):
            return web_search(action.split(':', 1)[1].strip())
        if action.startswith('calc:'):
            return calculator(action.split(':', 1)[1].strip())
        if "joke" in action.lower():
            # Dynamically generate a joke using Azure OpenAI with a safe prompt
            prompt = "Please tell me a short, family-friendly programming joke suitable for all audiences."
            response = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.9,
                max_tokens=60
            )
            content = response.choices[0].message.content
            joke = content.strip() if content is not None else ""
            return joke if joke else "Sorry, I couldn't come up with a joke right now."
        # Try to extract and evaluate a math expression from natural language
        import re
        match = re.search(r'result of ([\d\s\*\+\-/\.]+)', action.lower())
        if match:
            expr = match.group(1).strip()
            return calculator(expr)
        return f"Unknown action: {action}"

    def update(self, action: str, result: str):
        # Store the action-result pair as a new memory entry
        entry = f"Action: {action} -> Result: {result}"
        self.memory.add(entry)

    def run(self, steps: int = 5):
        # Main loop: repeat for a fixed number of steps or until goal achieved
        for i in range(steps):
            # 1) Build context (goal + last action + relevant memories)
            context = self.context_mgr.build(self.goal, self.last_action)
            # 2) Plan next action via the LLM
            action = self.planner.plan(context)
            print(f"Step {i+1} -> Planned action: {action}")
            # 3) Execute the planned action
            result = self.act(action)
            print(f"Step {i+1} -> Result: {result}\n")
            # 4) Update memory with the interaction
            self.update(action, result)
            # 5) Save action for next iteration's context
            self.last_action = action
            # 6) Simple stop condition: if action says 'done', break loop
            if 'done' in action.lower():
                print("Goal reached, exiting.")
                break

    def converse(self):
        """
        Interactive conversation loop with the LLM, storing context and memory.
        Ends only when the user types 'end conversation'.
        """
        print("\n=== Conversation started. Type 'end conversation' to stop. ===\n")
        conversation_history = []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "end conversation":
                print("Conversation ended.")
                break
            # Store user input in memory
            self.memory.add(f"User: {user_input}")
            # Build context from goal, last action, and relevant memories
            context = self.context_mgr.build(self.goal, self.last_action)
            # Add conversation history to context for continuity
            if conversation_history:
                context += "\nConversation so far:\n" + "\n".join(conversation_history)
            # LLM response
            system_message = "You are a helpful, safe, and professional AI assistant. Continue the conversation appropriately."
            response = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': context + "\nUser: " + user_input}
                ],
                temperature=0.7
            )
            content = response.choices[0].message.content
            agent_reply = content.strip() if content is not None else ""
            print(f"Agent: {agent_reply}\n")
            # Store agent reply in memory and conversation history
            self.memory.add(f"Agent: {agent_reply}")
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Agent: {agent_reply}")
            self.last_action = user_input  # For context continuity

# ------------------------
# 8. Basic Python Concepts Demo (Thoroughly Commented)
# ------------------------

# 8.1 Data Types & Variables
# --------------------------
# Variables store values. Below are examples of Python's basic types:
integer_var: int = 42               # Integer type for whole numbers
decimal_var: float = 3.1415         # Floating-point for decimals
string_var: str = "Hello, AI"     # String for sequences of characters
bool_var: bool = True               # Boolean for True/False values

# Compound data structures:
list_var: List[int] = [1, 2, 3, 4]   # List: ordered, changeable collection
tuple_var: tuple = ('a', 'b', 'c')   # Tuple: ordered, unchangeable collection
set_var: set = {1, 2, 3}             # Set: unordered collection of unique items

dict_var: Dict[str, int] = {'x': 1, 'y': 2}  # Dictionary: key-value pairs

# 8.2 Control Flow
# ----------------
# Use for loops to iterate over collections:
for num in list_var:
    # if-else statements let you branch logic:
    if num % 2 == 0:
        print(f"{num} is even")  # % is the modulo operator
    else:
        print(f"{num} is odd")

# while loops repeat as long as a condition holds true:
counter = 0                       # Initialize a counter
while counter < 3:                 # Loop while counter is less than 3
    print(f"Counting {counter}")
    counter += 1                  # Increment counter by 1 each time

# 8.3 List Comprehensions & Generators
# -------------------------------------
# List comprehension: concise way to create lists
squares = [x * x for x in range(10)]  # Creates a list of squares from 0 to 9

# Generator expression: similar syntax but produces items one at a time
generator_sq: Iterator[int] = (x * x for x in range(5))
for val in generator_sq:
    print("Generated square:", val)

# 8.4 Functions & Lambdas
# ------------------------
# Define reusable code blocks with def:
def greet(name: str = "World") -> str:
    """Return a greeting message for the provided name."""
    return f"Hello, {name}!"

print(greet("AI Engineer"))        # Call the function and print the result

# Lambdas: small anonymous functions for simple operations
double = lambda x: x * 2             # Defines a function that doubles its input
print("Double of 5 is", double(5))

# 8.5 Exception Handling
# -----------------------
# try-except-finally: handle errors gracefully without crashing
try:
    result = 10 / 0                # This will raise a ZeroDivisionError
    print(result)
except ZeroDivisionError as e:
    # This block runs if the above error occurs
    print("Error occurred:", e)
finally:
    # This always runs, even if there's an error
    print("Finished exception handling")

# 8.6 Decorators
# ---------------
# Decorators modify or enhance functions without changing their code

def debug(func):
    """A decorator that logs function calls and their results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@debug
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b

# Calling add will now print debug information:
add(3, 7)

# 8.7 Custom Context Manager
# ---------------------------
# Context managers ensure resources are set up and cleaned up properly
class open_file:
    """Simple context manager for file operations."""
    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode
    def __enter__(self):
        # Open the file and return the file object
        self.file = open(self.filename, self.mode)
        return self.file
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the file when done, even if an error occurred
        self.file.close()

# Using the custom context manager to write and then read a file:
with open_file('demo.txt', 'w') as f:
    f.write('Context managers are awesome!')
with open_file('demo.txt', 'r') as f:
    content = f.read()
    print(content)  # Prints the text we just wrote

# 8.8 Modules & Library Usage
# ----------------------------
# math module provides additional math functions:
print("Square root of 16 is", math.sqrt(16))  # sqrt computes square root

# Using random to pick a random element from a list:
print("Random item from list_var:", random.choice(list_var))

# ------------------------
# 9. Entry Point
# ------------------------
if __name__ == '__main__':
    print("=== Basic Concepts Demo Complete ===")
    goal = input("Enter a goal for the AI agent: ")
    agent = Agent(goal)
    # Start a conversation loop instead of the run() method
    agent.converse()

# To install the required packages, run the following command:
# pip install fastapi uvicorn requests beautifulsoup4

