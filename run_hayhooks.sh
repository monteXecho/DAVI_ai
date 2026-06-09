export HAYHOOKS_PIPELINES_DIR="$(pwd)/hayhooks_wrapper"
export HAYHOOKS_ADDITIONAL_PYTHONPATH="$(pwd)"
export PYTHONPATH="$(pwd):$PYTHONPATH"
uv run hayhooks run --host 0.0.0.0