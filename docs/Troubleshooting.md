% Troubleshooting

Pydantic UnsupportedFieldAttributeWarning
- Emitted by a dependency; harmless.
- Suppress if desired:
  - `import warnings; warnings.filterwarnings("ignore", message=".*validate_default.*")`

"libmagic is unavailable"
- Improves filetype detection for Unstructured.
- Windows: `uv pip install python-magic-bin`
- Debian/Ubuntu: `apt-get install libmagic1` and `uv pip install python-magic`

"'doc_id' is deprecated and 'id_' will be used instead"
- LlamaIndex info; safe to ignore.
- Suppress: `logging.getLogger("llama_index").setLevel(logging.ERROR)`
