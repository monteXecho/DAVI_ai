import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Union

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses.byte_stream import ByteStream

logger = logging.getLogger(__name__)


class DoclingMetaExtractor:
    """
    Class for extracting metadata from Docling documents.

    Provides default implementations that can be overridden by subclasses
    if custom metadata extraction is needed.
    """

    def __init__(self, filter_binary_hash: bool = True):
        """
        Initialize the DoclingMetaExtractor.

        :param filter_binary_hash: Whether to filter out binary_hash fields from metadata
        """
        self.filter_binary_hash = filter_binary_hash

    def _filter_binary_hash(self, data: dict) -> dict:
        """Recursively filter out binary_hash fields from metadata."""
        filtered = {}
        for key, value in data.items():
            if "binary_hash" in key and isinstance(value, int):
                continue
            if isinstance(value, dict):  # recursively process nested dictionaries
                filtered[key] = self._filter_binary_hash(value)
            else:
                filtered[key] = value
        return filtered

    def extract_dl_doc_meta(self, dl_doc: DoclingDocument) -> dict[str, Any]:
        """Extract metadata from a Docling document."""
        if not dl_doc.origin:
            return {}

        meta = {"dl_meta": {"origin": dl_doc.origin.model_dump(exclude_none=True)}}
        return self._filter_binary_hash(meta) if self.filter_binary_hash else meta


@component
class DoclingConverter:
    """
    Convert files to documents using Docling. `DoclingConverter` supports ByteStream input and serialization.

    Use DoclingConverter to convert files of different formats into Document objects using Docling.
    DoclingConverter accepts ByteStream objects as input and supports page break placeholders in Markdown exports.
    """

    def __init__(
        self,
        converter: Optional[DocumentConverter] = None,
        convert_kwargs: Optional[dict[str, Any]] = None,
        md_export_kwargs: Optional[dict[str, Any]] = None,
        meta_extractor: Optional[DoclingMetaExtractor] = None,
        filter_binary_hash: bool = True,
        page_break_placeholder: str = "\\f",
    ):
        """
        Create a Docling Haystack converter.

        :param converter: The Docling `DocumentConverter` instance to use; if not set, uses the default
            `DocumentConverter`.
        :param convert_kwargs: Additional keyword arguments to customize Docling conversion.
        :param md_export_kwargs: Additional keyword arguments to customize Markdown export.
        :param meta_extractor: The `DoclingExtractor` instance to use for populating the output document metadata.
        :param filter_binary_hash: Whether to filter out binary_hash fields from document metadata.
        :param page_break_placeholder: The string to use as a page break placeholder in the Markdown export.
        """
        self._converter = converter or DocumentConverter()
        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}

        # init md_export_kwargs with defaults if not provided
        if md_export_kwargs is None:
            md_export_kwargs = {"image_placeholder": ""}  # keep out images by default

        # add page_break_placeholder to md_export_kwargs
        md_export_kwargs["page_break_placeholder"] = page_break_placeholder
        self._md_export_kwargs = md_export_kwargs

        self._meta_extractor = meta_extractor or DoclingMetaExtractor(
            filter_binary_hash=filter_binary_hash
        )
        self.page_break_placeholder = page_break_placeholder
        self.filter_binary_hash = filter_binary_hash  # store for serialization

        # store initialization parameters for serialization
        self._init_params = {
            "convert_kwargs": convert_kwargs,
            "md_export_kwargs": md_export_kwargs,
            "filter_binary_hash": filter_binary_hash,  # add to stored params
            "page_break_placeholder": page_break_placeholder,
        }

    def _extract_filename_with_extension(
        self, source: Union[str, Path, ByteStream], bytestream: ByteStream
    ) -> str:
        """Extract the filename with extension from the source or bytestream."""
        filename = "document.bin"  # generic default

        # get filename from source
        source_filename = self._get_filename_from_source(source)
        if source_filename:
            filename = source_filename

        # fallback: get filename from bytestream metadata
        bytestream_filename = self._get_filename_from_meta(
            bytestream.meta if hasattr(bytestream, "meta") else {}
        )
        if bytestream_filename:
            filename = bytestream_filename

        return filename

    def _get_filename_from_source(
        self, source: Union[str, Path, ByteStream]
    ) -> Optional[str]:
        """Extract filename from a source object."""
        if isinstance(source, (str, Path)):
            return str(Path(source).name)
        if isinstance(source, ByteStream) and hasattr(source, "meta"):
            return self._get_filename_from_meta(source.meta)
        return None

    def _get_filename_from_meta(self, meta: dict) -> Optional[str]:
        """Extract filename from metadata dictionary."""
        filename = meta.get("file_name") or meta.get("name")
        if not filename:
            return None

        # get extension from metadata if not already present in filename
        if not Path(filename).suffix and "mime_type" in meta:
            extension = mimetypes.guess_extension(meta["mime_type"])
            if extension:
                return f"{Path(filename).stem}{extension}"

        return filename

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: dict[str, Any] | List[dict[str, Any]] | None = None,
    ):
        """
        Run the DoclingConverter.

        :param sources: The list of file paths, Path objects, or ByteStream objects to convert.
        :param meta: Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources, as they are zipped together.
        :return: Dictionary with key "documents" containing the output Haystack Documents.
        """
        if any(
            (isinstance(source, (str, Path)) and str(source).lower().endswith(".docx"))
            or (
                isinstance(source, ByteStream)
                and source.meta.get("file_extension", "").lower() == "docx"
            )
            for source in sources
        ):
            logger.warning(
                "You are processing one or more DOCX files. "
                "Page numbers cannot be reliably extracted from DOCX due to format limitations "
                "(see https://github.com/docling-project/docling/discussions/997). "
                "If you require page number support, please convert your DOCX files to PDF before ingestion."
            )

        meta_list = normalize_metadata(meta, sources_count=len(sources))
        documents: list[Document] = []

        # process each source separately
        try:
            for source, metadata in zip(sources, meta_list, strict=False):
                try:
                    bytestream = get_bytestream_from_source(source)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(f"Could not read {source}. Skipping it. Error: {e}")
                    continue
                # extract filename with extension
                filename = self._extract_filename_with_extension(source, bytestream)
                # create a DocumentStream
                source_docling = DocumentStream(
                    name=filename,
                    stream=BytesIO(bytestream.data),
                )
                # convert the DocumentStream using chosen Docling DocumentConverter
                dl_doc = self._converter.convert(
                    source=source_docling,
                    **self._convert_kwargs,
                ).document

                # extract metadata from the Docling document
                doc_meta = self._meta_extractor.extract_dl_doc_meta(dl_doc=dl_doc)
                doc_meta.update(bytestream.meta)
                doc_meta.update(metadata)

                # turn the Docling document into a Haystack Document
                hs_doc = Document(
                    content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                    meta=doc_meta,
                )
                documents.append(hs_doc)  # add to output documents list
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Unexpected error in DoclingConverter.run: {e}")
        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary for pipeline persistence.

        :return: A dictionary representation of the component
        """
        return default_to_dict(
            self,
            convert_kwargs=self._convert_kwargs,
            md_export_kwargs=self._md_export_kwargs,
            filter_binary_hash=self.filter_binary_hash,
            page_break_placeholder=self.page_break_placeholder,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoclingConverter":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary representation of the component
        :return: A new instance of the component
        """
        return default_from_dict(cls, data)
