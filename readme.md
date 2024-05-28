# Agent with Ollama & Langchain

You can test this app in lightning studios via this [link](https://lightning.ai/maxidiazbattan/studios/grateful-sapphire-bo61), or locally following the steps below.

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
ollama pull llama3
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
