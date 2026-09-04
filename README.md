# News Simplification Assistant

This project fine-tunes OPT-125M, OPT-1.3B, and OPT-2.7B to simplify news
articles and generate seven structured aspects. The Flask application performs
inference locally and loads one selected OPT model at a time.

## Application workflow

1. The user enters a news article and selects a model.
2. The application cleans and validates the article.
3. TF-IDF extracts up to eight important source terms.
4. The prompt service combines the instructions, source terms, and article.
5. The model manager loads the selected base model and optional LoRA adapter.
6. The model generates a simplified summary and seven aspects.
7. The output parser returns the structured result to the Flask interface.

## Supported models

| Model ID | Display name | Base model | Local adapter |
|---|---|---|---|
| `opt125m_base` | Original OPT-125M | `facebook/opt-125m` | Disabled |
| `opt125m_news` | Fine-tuned OPT-125M | `facebook/opt-125m` | `models/opt-125m-news-model` |
| `opt13b_base` | Original OPT-1.3B | `facebook/opt-1.3b` | Disabled |
| `opt13b_news` | Fine-tuned OPT-1.3B | `facebook/opt-1.3b` | `models/opt-1.3b-news-model` |
| `opt27b_base` | Original OPT-2.7B | `facebook/opt-2.7b` | Disabled |
| `opt27b_news` | Fine-tuned OPT-2.7B | `facebook/opt-2.7b` | `models/opt-2.7b-news-model` |

The application downloads the original OPT weights from Hugging Face when they
are not already cached. Each fine-tuned option combines the corresponding base
model with its local LoRA adapter.

## Local setup

Create and activate a Python environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Local inference requires enough RAM or GPU memory for the selected model.
The application uses 4-bit loading when CUDA and bitsandbytes are available and
otherwise falls back to CPU-compatible loading.

Start the application:

```powershell
flask --app app run
```

Open `http://127.0.0.1:5000` in a browser.

## Active deployment files

- `app.py`: Flask routes and response handling
- `services/model_manager.py`: model loading, switching, and generation
- `services/prompt_service.py`: prompt construction and TF-IDF guidance
- `services/preprocessor.py`: article cleaning and length checks
- `services/output_parser.py`: summary and aspect parsing
- `templates/index.html`: web interface
- `static/style.css`: interface styling
