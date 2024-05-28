
# tools
import os
import glob

# utils
from utils.utils import DocumentToolsGenerator
from utils.prompts import context

# llamaindex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent

# warnings removal
from warnings import filterwarnings
filterwarnings('ignore')


path = './data'
file_name = 'shap.pdf'
file_path = os.path.join(path, file_name)


models = {
  'mistral': 'koesn/mistral-7b-instruct',
  'misal': 'smallstepai/misal-7B-instruct-v0.1',
  'codellama': 'codellama'
  }

# global settings
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model=models['mistral']) 

Settings.embed_model = embed_model
Settings.llm = llm

# documentToolsGenerator class instantiation 
docs_tools = DocumentToolsGenerator(file_path=file_path)

# nodes creation
nodes = docs_tools.data_ingestion()

# tool generation
vector_tool, summary_tool, file_tool = docs_tools.tool_generator(nodes=nodes)

# agent initialization 
agent = ReActAgent.from_tools(tools=[vector_tool, summary_tool, file_tool], llm=llm, context=context, verbose=True)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
  result = agent.query(prompt)
  print(result)

