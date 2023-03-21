from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
from IPython.display import Markdown, display

"""
    augment LLM with reddit data.
    in-context learning: insert context into input prompt. Generate answers to our questions
    How to deal with Input Text token limit?

    LlamaIndex also can help connect many external datasources to your LLM

    Use Package LlamaIndex to:
    1. Create an index of text chunks from context
    2. When user asks a question, find most relevant chunks
    3. Answer user's question using the most relevant chunks as context

    """