# QuestionGuard

QuestionGuard is a Streamlit application for generating questions and checking them against a question corpus for exact, likely semantic, or possible duplicates.

## Run locally on Windows

Use Python 3.11 or 3.12 for straightforward dependency compatibility.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Run these commands from the folder containing `app.py` and `requirements.txt`.

## Data storage

The app stores its question corpus and decision history in `questions.db` (SQLite), if that is the database path configured in `app.py`. Back up this file if you need to preserve your data.

## Gemini API key

If Gemini-powered generation is enabled in your app, configure the API key using the environment variable or secrets method expected by your `app.py`. Never commit API keys to Git or put them in a public repository.

## Before public deployment

- Configure API keys through the hosting provider's secret settings.
- Confirm database persistence. Some hosts do not preserve local SQLite files between redeployments.
- Do not publish private question banks or personal data.
- Check that the installed packages match the imports used by your current `app.py`.
