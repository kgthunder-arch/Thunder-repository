# Common Ground Flask App

This project now runs as a Flask app with two experiences:

- `/` serves the new Common Ground global collaboration product experience
- `/agribusiness` preserves the existing agribusiness predictive analysis workflow

## Main Files

- `app.py`: Flask entrypoint and routing
- `templates/common_ground.html`: Common Ground homepage
- `static/common-ground/styles.css`: Common Ground visual design
- `static/common-ground/app.js`: Common Ground interactions
- `templates/index.html`, `templates/configure.html`, `templates/results.html`: agribusiness workflow

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

- `http://127.0.0.1:5000/` for Common Ground
- `http://127.0.0.1:5000/agribusiness` for the agribusiness analyzer

## Notes

- In this Codex environment, `python` is not currently available on the PATH, so I could not launch the Flask server here.
- The Flask wiring, templates, and static assets are in place and ready to run on a machine with Python installed.
