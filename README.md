# Fake News Detector (Hindi + English)

Flask + scikit-learn web app that classifies news as **Real** or **Fake** with a confidence score. Supports English and Hindi (Devanagari) text.

## Features
- TF-IDF + PassiveAggressiveClassifier (calibrated for probability)
- Hindi + English support (character + word n-grams)
- Clean dark-themed UI, mobile responsive
- REST endpoint `/api/predict` returning JSON
- Portfolio & GitHub ready

## Quickstart

```bash
pip install -r requirements.txt
python train.py        # trains on the bundled sample dataset, writes model/model.joblib
python app.py          # runs on http://localhost:5000
```

## Project structure
```
.
├── app.py              # Flask server
├── train.py            # Train script (scikit-learn)
├── requirements.txt
├── data/
│   └── sample_news.csv # tiny demo dataset (replace with Kaggle Fake/Real + BBC Hindi)
├── model/
│   └── model.joblib    # generated after running train.py
├── templates/
│   └── index.html
└── static/
    └── styles.css
```

## Recommended datasets
- English: [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Hindi: [Hindi Fake News Dataset](https://www.kaggle.com/datasets/anushkaml/hindi-fake-news-dataset)

Combine them into one CSV with columns `text,label` (label: `real` or `fake`).

## Deploy
- Render / Railway / Fly.io: standard Python web service, `python app.py` or `gunicorn app:app`
- Procfile included for Heroku-style platforms
