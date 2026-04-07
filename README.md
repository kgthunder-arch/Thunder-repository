# Agribusiness Predictive Analysis App

This Flask app lets agribusiness teams upload crop, weather, and market datasets, compare multiple machine learning models, and review saved forecast analyses.

## Main Files

- `app.py`: Flask entrypoint, model training flow, and storage integration
- `api/index.py`: Vercel Python function entrypoint
- `templates/index.html`, `templates/configure.html`, `templates/results.html`: agribusiness workflow UI
- `static/css/styles.css`, `static/js/app.js`: frontend styles and chart rendering
- `sample_data/agribusiness_sample.csv`: bundled demo dataset

## Run Locally

1. Create and activate a virtual environment:

```powershell
cd C:\Users\chans\Downloads\common-ground-app
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the server:

```powershell
python app.py
```

4. Open:

- `http://127.0.0.1:5000/`

## Storage Modes

- Local development:
  uploaded datasets and saved analyses are stored in `storage/`
- Vercel with `BLOB_READ_WRITE_TOKEN`:
  uploads and saved analyses are stored durably in Vercel Blob
- Vercel without `BLOB_READ_WRITE_TOKEN`:
  uploads use temporary runtime storage only

## Notes

- Saved analyses can be reopened from the homepage history section.
- `storage/` is ignored by Git so local saved uploads and reports stay out of the repository.
