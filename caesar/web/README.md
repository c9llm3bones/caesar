# Caesar Web UI

Run from the repository root:

```bash
CAESAR_CHECKPOINT=/path/to/checkpoint.pt \
CAESAR_CONFIG=small_inner_atom37_main_sc \
CAESAR_DEVICE=cuda \
python -m uvicorn caesar.web.app:app --host 0.0.0.0 --port 8000
```

Install web dependencies with:

```bash
pip install -e '.[web]'
```
