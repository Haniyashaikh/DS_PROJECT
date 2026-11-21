# Electricity Consumption Analysis and Prediction System

Bahria University Karachi Campus - Introduction to Data Science Lab

Analysis and Zone 1 prediction on the *Power Consumption of Tetouan City* dataset
(UCI Machine Learning Repository, Salam & El Hibaoui, 2018, CC BY 4.0).
This is a demonstration on Moroccan data, not a prediction for Karachi or Pakistan.

## Folder structure

```
project/
├── data/Tetuan City power consumption.csv   <- you download this
├── models/                                  <- created by train_model.py
├── outputs/                                 <- created by eda.py (charts)
├── common.py        shared loading + cleaning code
├── eda.py           exploratory data analysis (charts)
├── train_model.py   trains and saves the models
├── app.py           Streamlit dashboard
└── requirements.txt
```

## How to run (PyCharm terminal)

1. Create the `data` folder and put the CSV inside it.
   Download: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city
2. Install libraries:  `pip install -r requirements.txt`
3. Charts and insights:  `python eda.py`
4. Train models:  `python train_model.py`
5. Start dashboard:  `streamlit run app.py`

## Deployment (Streamlit Community Cloud)

1. Push everything (including `data/` and `models/`) to GitHub.
2. Go to https://share.streamlit.io, connect the repo, choose `app.py`, deploy.
