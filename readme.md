# Agent with Ollama & Langchain

This repo contains an implementation of an Agentic Retrieval-Augmented Generation (RAG) application leveraging LlamaIndex and Ollama. The agent efficiently retrieves relevant information from documents and generates coherent, contextually appropriate responses, thanks to the tools provided. Additionally, you can save the responses to a file if desired.
If you want, you can test this app in lightning studios via this [link](https://lightning.ai/maxidiazbattan/studios/agentic-rag-llamaindex-ollama), or locally following the steps below.

### 1. [Install](https://github.com/ollama/ollama?tab=readme-ov-file) ollama and pull models

On linux
```shell
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama

```shell
ollama serve
```

Pull the LLM you'd like to use:

```shell
ollama pull koesn/mistral-7b-instruct
```

### 2. Create a virtual environment

```shell
python -m venv venv
source venv/bin/activate
```

### 3. Install libraries

```shell
pip install -r requirements.txt
```

### 4. Download the pdf to RAG, in this case a paper about Shap values
```shell
wget https://arxiv.org/pdf/2302.08160 -O ./data/shap.pdf
```
