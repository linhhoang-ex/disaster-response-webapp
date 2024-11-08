# Disaster Response Pipeline Project

## Introduction
This project analyze the [disaster response dataset](https://huggingface.co/datasets/community-datasets/disaster_response_messages) from [Appen](https://www.appen.com/) to build a model for an API that classifies disaster messages. The dataset contains real messages that were sent during disaster events. A machine learning pipeline is developed to categorize these events so that one can send the messages to an appropriate disaster relief agency.

This project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Bellow are a few screenshots of the web app.

<p align="center">
  <kbd>
  <img src="/app/static/img/screenshot-1-dataset-overview.jpeg" alt="screenshot-1-dataset-overview" style="border:3px solid #ffffff"><br>
  </kbd>
  <em>Some statistics of the dataset</em>
</p>

<p align="center">
  <kbd>
  <img src="/app/static/img/screenshot-2-classification-example.jpeg" alt="screenshot-1-dataset-overview" style="border:3px solid #000000"><br>
  </kbd>
  <em>An example of classifying a disaster response message</em>
</p>

## About the dataset
The [disaster response dataset](https://huggingface.co/datasets/community-datasets/disaster_response_messages) contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety. Upon release, this is the featured dataset of a new Udacity course on Data Science and the AI4ALL summer school and is especially utile for text analytics and natural language processing (NLP) tasks and models.The input data in this job contains thousands of untranslated disaster-related messages and their English translations.

## Structure of the repository
```
├── app/
│   ├── static/img
│   ├── templates/
│   │   ├── master.html  # main page of web app
│   │   └── go.html  # classification result page of web app
│   └── run.py  # script that runs the webapp using Flask
├── data/
│   ├── disaster_categories.csv  # data to be processed: message categories
│   ├── disaster_messages.csv  # data to be processed: disaster response messages
│   ├── DisasterResponse.db   # cleaned data will be exported to this SQL database
│   └── process_data.py    # ETL pipeline for data cleaning
├── models/
│   ├── train_classifier.py  # NLP pipeline for training a text-based classifier
│   └── classifier.pkl  # trained model
├── README.md
└── environment.yml   # dependencies of the conda environment
```

## How to run the web application
0. Create a conda environment with dependencies specified in [environment.yml](environment.yml).
1. Run the following commands in the project's root directory to set up the database and train a classification model.
   - To run ETL pipeline that cleans data and stores in database:
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains a classifier and saves the trained model to a pickle file:
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`.
3. Run your web app: `python run.py`.
4. Access `127.0.0.1:3000` in a web browser (e.g., Google Chrome) to open the webapp's homepage.
