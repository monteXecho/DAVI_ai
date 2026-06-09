import argparse
import os
import subprocess

from pathlib import Path

from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import (
    DocumentSplitter,
    RecursiveDocumentSplitter,
)
from haystack.components.writers import DocumentWriter
from haystack.components.extractors.llm_metadata_extractor import LLMMetadataExtractor
from haystack.components.joiners.document_joiner import DocumentJoiner

from haystack.components.converters import (
    PDFMinerToDocument,
    DOCXToDocument,
    HTMLToDocument
)
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.routers import FileTypeRouter

from haystack.utils import ComponentDevice, Device

from loguru import logger

from davi.components.docling_converter import DoclingConverter
from davi.components.markdown_splitter import MarkdownHeaderSplitter
from davi.document_store import DOC_STORE_TYPE, INDEX_NAME, INDEX_PATH, main_doc_store
from davi.llm import create_chat_generator
from davi.prompts import CHUNK_INDEX_PROMPT


try:
    subprocess.check_output('nvidia-smi')
    RUN_DEVICE = ComponentDevice.from_single(Device.gpu(id=0))
except FileNotFoundError:
    RUN_DEVICE = None


INDEXING_PIPELINE = os.environ.get("INDEXING_PIPELINE", "indexing_pipeline_v1")


def _indexing_pipeline_v1(doc_store):
    converter = PDFMinerToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=RUN_DEVICE,
    )
    writer = DocumentWriter(doc_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("converter", "splitter")
    indexing_pipeline.connect("splitter", "cleaner")
    indexing_pipeline.connect("cleaner", "doc_embedder")
    indexing_pipeline.connect("doc_embedder", "writer")

    return indexing_pipeline


def _indexing_pipeline_v2(doc_store):
    converter = DoclingConverter()
    cleaner = DocumentCleaner()
    splitter = RecursiveDocumentSplitter(
        split_length=500,
        split_overlap=100,
        split_unit="word",
        separators=["\n\n", "sentence", "\n", " "],
        sentence_splitter_params={
            "language": "nl",
            "extend_abbreviations": False,
        },
    )
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="BAAI/bge-m3",
        device=RUN_DEVICE,
    )
    writer = DocumentWriter(doc_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("converter", "splitter")
    indexing_pipeline.connect("splitter", "cleaner")
    indexing_pipeline.connect("cleaner", "doc_embedder")
    indexing_pipeline.connect("doc_embedder", "writer")

    return indexing_pipeline


def _indexing_pipeline_v3(doc_store):
    converter = DoclingConverter()
    cleaner = DocumentCleaner()
    splitter = RecursiveDocumentSplitter(
        split_length=300,
        split_overlap=20,
        split_unit="word",
        separators=["\n\n", "sentence", "\n", " "],
        sentence_splitter_params={
            "language": "nl",
            "extend_abbreviations": False,
        },
    )
    markdown_splitter = MarkdownHeaderSplitter(
        split_length=500, split_overlap=20
    )
    doc_embedder = SentenceTransformersDocumentEmbedder(
        # model="BAAI/bge-m3",
        # model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model="intfloat/multilingual-e5-large",
        device=RUN_DEVICE,
    )
    writer = DocumentWriter(doc_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("markdown_splitter", markdown_splitter)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("converter", "markdown_splitter")
    indexing_pipeline.connect("markdown_splitter", "splitter")
    indexing_pipeline.connect("splitter", "cleaner")
    indexing_pipeline.connect("cleaner", "doc_embedder")
    indexing_pipeline.connect("doc_embedder", "writer")

    return indexing_pipeline

def _indexing_pipeline_v4(doc_store):
    file_classifier = FileTypeRouter(
        mime_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/html",
        ]
    )
    pdf_converter = PDFMinerToDocument()
    docx_converter = DOCXToDocument()
    html_converter = HTMLToDocument(
        extraction_kwargs={
            "output_format": "markdown",
            "target_language": None,
            "include_tables": True,
            "include_links": True,
        }
    )
    joiner = DocumentJoiner(join_mode="concatenate", sort_by_score=False)
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=400,
        split_overlap=40,
        respect_sentence_boundary=True,
        language="nl",
        extend_abbreviations=False,
    )
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="google/embeddinggemma-300m",
        device=RUN_DEVICE,
        trust_remote_code=True,
        token=Secret.from_env_var("HF_TOKEN"),
        batch_size=8,
    )
    writer = DocumentWriter(doc_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("file_classifier", file_classifier)
    indexing_pipeline.add_component("pdf_converter", pdf_converter)
    indexing_pipeline.add_component("docx_converter", docx_converter)
    indexing_pipeline.add_component("html_converter", html_converter)
    indexing_pipeline.add_component("joiner", joiner)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("file_classifier.application/pdf", "pdf_converter.sources")
    indexing_pipeline.connect(
        "file_classifier.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_converter.sources",
    )
    indexing_pipeline.connect(
        "file_classifier.text/html", "html_converter.sources"
    )
    indexing_pipeline.connect("pdf_converter.documents", "joiner.documents")
    indexing_pipeline.connect("docx_converter.documents", "joiner.documents")
    indexing_pipeline.connect("html_converter.documents", "joiner.documents")
    indexing_pipeline.connect("joiner", "splitter")
    indexing_pipeline.connect("splitter", "cleaner")
    indexing_pipeline.connect("cleaner", "doc_embedder")
    indexing_pipeline.connect("doc_embedder", "writer")

    return indexing_pipeline


def _indexing_pipeline_v5(doc_store):
    converter = PDFMinerToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=1200,
        split_overlap=40,
        respect_sentence_boundary=True,
        language="nl",
        extend_abbreviations=False,
    )
    extractor = LLMMetadataExtractor(
        prompt=CHUNK_INDEX_PROMPT,
        chat_generator=create_chat_generator(),
        expected_keys=["desc"],
        raise_on_failure=False,
    )
    final_splitter = DocumentSplitter(
        split_by="word",
        split_length=400,
        split_overlap=40,
        respect_sentence_boundary=True,
        language="nl",
        extend_abbreviations=False,
    )

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="google/embeddinggemma-300m",
        device=RUN_DEVICE,
        trust_remote_code=True,
        token=Secret.from_env_var("HF_TOKEN"),
        meta_fields_to_embed=["desc"]
    )
    writer = DocumentWriter(doc_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("extractor", extractor)
    indexing_pipeline.add_component("final_splitter", final_splitter)
    indexing_pipeline.add_component("joiner", DocumentJoiner())
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("converter", "splitter")
    indexing_pipeline.connect("splitter", "cleaner")
    indexing_pipeline.connect("cleaner", "extractor")
    indexing_pipeline.connect("extractor.documents", "joiner")
    indexing_pipeline.connect("extractor.failed_documents", "joiner")
    indexing_pipeline.connect("joiner", "final_splitter")
    indexing_pipeline.connect("final_splitter", "doc_embedder")
    indexing_pipeline.connect("doc_embedder", "writer")

    return indexing_pipeline



def create_indexing_pipeline(doc_store, pipeline_name):
    if pipeline_name == "indexing_pipeline_v1":
        return _indexing_pipeline_v1(doc_store)
    if pipeline_name == "indexing_pipeline_v2":
        return _indexing_pipeline_v2(doc_store)
    if pipeline_name == "indexing_pipeline_v3":
        return _indexing_pipeline_v3(doc_store)
    if pipeline_name == "indexing_pipeline_v4":
        return _indexing_pipeline_v4(doc_store)
    if pipeline_name == "indexing_pipeline_v5":
        return _indexing_pipeline_v5(doc_store)
    raise ValueError(f"Unknown indexing pipeline {pipeline_name}")


def index_data(data_path, indexing_pipeline):
    file_paths = list(Path(data_path).glob("**/*.pdf"))
    logger.info(f"Found {len(file_paths)} files to index.")
    if len(file_paths) == 0:
        logger.warning(f"No files to index in {data_path}")
        return
    indexing_pipeline.run({"converter": {"sources": file_paths}})


def index_data_with_pipeline(data_path, pipeline_name):
    indexing_pipeline = create_indexing_pipeline(
        doc_store=main_doc_store, pipeline_name=pipeline_name
    )

    index_data(data_path, indexing_pipeline)
    if DOC_STORE_TYPE == "InMemory":
        main_doc_store.save_to_disk(INDEX_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create index from a data directory.")
    parser.add_argument(
        "-d", "--data_directory", required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "-p",
        "--pipeline_name",
        choices=[
            "indexing_pipeline_v1",
            "indexing_pipeline_v2",
            "indexing_pipeline_v3",
            "indexing_pipeline_v4",
            "indexing_pipeline_v5",
        ],
        default="indexing_pipeline_v1",
        help="Name of the pipeline to run (default: indexing_pipeline_v1)",
    )

    args = parser.parse_args()

    logger.info(f"Data directory provided: {args.data_directory}")
    index_data_with_pipeline(
        data_path=args.data_directory,
        pipeline_name=args.pipeline_name,
    )
