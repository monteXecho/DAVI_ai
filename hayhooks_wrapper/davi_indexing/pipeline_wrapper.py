import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from typing import List, Optional
from fastapi import UploadFile
import json
import tempfile
from pathlib import Path


from hayhooks import BasePipelineWrapper


from davi.document_store import main_doc_store, get_document_store
from davi.document_deletion import delete_indexed_documents_by_file_ids
from davi.indexing import create_indexing_pipeline


class PipelineWrapper(BasePipelineWrapper):

    def setup(self) -> None:
        self.pipeline = create_indexing_pipeline(
            doc_store=main_doc_store,
            pipeline_name='indexing_pipeline_v4',
        )

    def run_api(
        self,
        index_id: Optional[str] = None,
        files: List[UploadFile] = None,
        original_file_paths: List[str] = None,
        file_ids: List[str] = None,
        files_meta_data: List[str] = None,
    ) -> dict:
        try:
            if (files is not None) and (file_ids is not None):
                if len(files) != len(file_ids):
                    return {"message": f"Error: {len(files)=} != {len(file_ids)=}"}
                files_meta_data_dicts = []
                if files_meta_data is not None:
                    if (len(files) != len(files_meta_data)):
                        return {"message": f"Error: files_meta_data provided but {len(files)=} != {len(files_meta_data)=}"}
                    for file_meta_data_json_str in files_meta_data:
                        try:
                            files_meta_data_dicts.append(json.loads(file_meta_data_json_str))
                        except json.decoder.JSONDecodeError as e:
                            return {"message": f"Error parsing files_meta_data json str: {file_meta_data_json_str}"}
                with tempfile.TemporaryDirectory() as temp_dir:
                    if index_id is not None:
                        writer = self.pipeline.get_component('writer')
                        doc_store = get_document_store(index_name=index_id)
                        writer.document_store = doc_store
                    else:
                        doc_store = main_doc_store

                    # Drop stale chunks for the same file_ids (re-upload after DB-only delete)
                    delete_indexed_documents_by_file_ids(doc_store, file_ids)

                    file_paths = []
                    metas = []
                    if original_file_paths is None:
                        original_file_paths = [None] * len(files)
                    for i, (file, file_id, original_file_path) in enumerate(zip(files, file_ids, original_file_paths)):
                        file_path = Path(temp_dir) / file.filename
                        file_path.write_bytes(file.file.read())
                        file_paths.append(file_path)
                        file_meta_data = {'file_id': file_id, 'original_file_path': original_file_path}
                        if len(files_meta_data_dicts) > 0:
                            file_meta_data.update(files_meta_data_dicts[i])
                        metas.append(file_meta_data)
                    self.pipeline.run({"file_classifier": {"sources": file_paths, 'meta': metas}})

                return {
                    "message": f"Files indexed successfully: {[file.filename for file in files]}"
                }
            else:
                return {"message": "No files+file_ids provided"}
        except Exception as e:
            return {"message": f"Error indexing files: {e}"}