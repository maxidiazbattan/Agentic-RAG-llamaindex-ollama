# tools
import os
from typing import Optional, List, Tuple

# utils
from utils.tools import save_file

# llamaindex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.tools import FunctionTool,QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition


class DocumentToolsGenerator:
    """
    Document processing and tool generation for vector search and summarization.

    """

    def __init__(self, file_path: str):
        """
        Initializes the DocumentToolsGenerator with the given file path.

        Args:
            file_path (str): The path to the document file to be processed.
        """
        self.file_path = file_path
        self.vector_index = None

    def data_ingestion(self, chunk_size: int = 1024, chunk_overlap: int = 64) -> List[BaseNode]:
        """
        Loads and splits the document into chunks for processing.

        Args:
            chunk_size (int): The size of each chunk. Default is 1024.
            chunk_overlap (int): The overlap between chunks. Default is 64.

        Returns:
            List[BaseNode]: A list of document nodes created from the chunks.
        """
        documents = SimpleDirectoryReader(input_files=[self.file_path]).load_data()
        sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = sentence_splitter.get_nodes_from_documents(documents=documents)

        return nodes

    def vector_query(self, query: str, page_numbers: Optional[List[str]] = None) -> str:
        """
        Performs a vector search over the document using the specified query and optional page numbers.

        Args:
            query (str): The query string to be embedded for the search.
            page_numbers (Optional[List[str]]): A list of page numbers to be retrieved. Defaults to searching over all pages.

        Returns:
            str: The search response.
        """
        page_numbers = page_numbers or []
        metadata_dict = [{"key": 'page_label', "value": p} for p in page_numbers]
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dict,
                condition=FilterCondition.OR)
        )
        response = query_engine.query(query)
        return response

    def tool_generator(self, nodes: List[BaseNode], vector_store_path: str = './data/db', db_name: str = 'vs') -> Tuple[FunctionTool, QueryEngineTool, FunctionTool]:
        """
        Generates and returns tools for vector search, document summarization, and file saving.

        Args:
            nodes (List[BaseNode]): The list of nodes generated from the document.
            vector_store_path (str): The path to store the vector index. Default is './data/db'.
            db_name (str): The name of the vector store database. Default is 'vs'.

        Returns:
            Tuple[FunctionTool, QueryEngineTool, FunctionTool]: A tuple containing the vector query tool, summary query tool, and file saving tool.
        """
        # vector index
        if not os.path.exists(db_name):
            self.vector_index = VectorStoreIndex(nodes=nodes)
            self.vector_index.storage_context.vector_store.persist(persist_path=f'{vector_store_path}/{db_name}')
        else:
            self.vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f'{vector_store_path}')
            )

        # summary index
        summary_index = SummaryIndex(nodes=nodes)

        # prepare vector tool
        vector_query_tool = FunctionTool.from_defaults(
            name="vector_search_tool", 
            fn=self.vector_query,
            description="Useful for searching specific facts in a document"
        )

        # prepare summary tool
        summary_query = summary_index.as_query_engine(response_mode="tree_summarize")
        summary_query_tool = QueryEngineTool.from_defaults(
            name="summary_query_tool",
            query_engine=summary_query,
            description="Useful for summarizing an entire document. DO NOT USE if you have specified questions over the documents."
        )

        # prepare file saving tool
        file_tool = FunctionTool.from_defaults(
            name="file_saver_tool",
            fn=save_file,
            description="Useful for saving a text file"
        )

        return vector_query_tool, summary_query_tool, file_tool
