# Central inference deployment

The laptop Flask application calls one Modal T4 endpoint. The endpoint loads
the OPT-125M, OPT-1.3B, and OPT-2.7B model families in 4-bit mode and supports
six request model IDs:

- `opt125m_base`: original OPT-125M with the LoRA adapter disabled.
- `opt125m_news`: OPT-125M with the news adapter enabled.
- `opt13b_base`: original OPT-1.3B with the LoRA adapter disabled.
- `opt13b_news`: OPT-1.3B with the news adapter enabled.
- `opt27b_base`: original OPT-2.7B with the LoRA adapter disabled.
- `opt27b_news`: the same base with the news adapter enabled.

The adapters are loaded from the private Hugging Face repository
`hehehehe6625/news-simplication`, subfolders `opt-125m`, `opt-1.3b`, and
`opt-2.7b`.

## Upload the OPT-125M adapter

```powershell
.\.venv\Scripts\hf.exe upload hehehehe6625/news-simplication .\models\opt-125m-news-model opt-125m --exclude "checkpoints/**" --commit-message "Add OPT-125M news adapter"
```

## Upload the OPT-1.3B adapter

```powershell
.\.venv\Scripts\hf.exe upload hehehehe6625/news-simplication .\models\opt-1.3b-news-model opt-1.3b --exclude "checkpoints/**" --commit-message "Add OPT-1.3B news adapter"
```

## Required Modal secrets

Create these in the Modal dashboard:

- `huggingface-secret` with key `HF_TOKEN`.
- `news-api-secret` with key `API_TOKEN`.

Never commit either value.

## Warm the model cache

Run this after initially uploading a model or replacing its files:

```powershell
.\.venv\Scripts\modal.exe run cloud\modal_inference.py::download_models
```

## Deploy

```powershell
.\.venv\Scripts\modal.exe deploy cloud\modal_inference.py
```

## Test through Modal authentication

```powershell
.\.venv\Scripts\modal.exe run cloud\modal_inference.py --model opt125m_news
.\.venv\Scripts\modal.exe run cloud\modal_inference.py --model opt125m_base
.\.venv\Scripts\modal.exe run cloud\modal_inference.py --model opt13b_news
.\.venv\Scripts\modal.exe run cloud\modal_inference.py --model opt13b_base
.\.venv\Scripts\modal.exe run cloud\modal_inference.py --model opt27b_news
.\.venv\Scripts\modal.exe run cloud\modal_inference.py --model opt27b_base
```

## Configure a laptop

Copy `.env.example` to `.env`, then replace the placeholder token with the
same value stored as `API_TOKEN` in `news-api-secret`:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by Git. Start Flask normally after saving it.
