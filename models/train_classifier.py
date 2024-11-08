import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')


def load_data(database_filepath: str) -> tuple:
    '''
    Load data from the SQLite database.

    Params
    ------
    - `database_filepath`: filepath to the SQLite database

    Return
    ------
    - `X`: all disaster response messages in the database
    - `Y`: the categories of each message in the database
    - `col_names`: names of the categories (36 entries)
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessage', con=engine)
    X = df['message']
    Y = df[df.columns.to_list()[4:]]
    col_names = df.columns[4:].to_list()

    return X, Y, col_names


def tokenize(text: str) -> list[str]:
    '''
    Tokenize: break a message (may include several sentences) into tokens.

    Params
    ------
    - `text`: the input message

    Return
    ------
    - `tokens`: the list of tokens for the input message
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text.lower())

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words]

    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    A transformer that extracts the part-of-speech of the first token for a
    given input message.
    '''

    def starting_verb(self, text: str):
        ''''
        Extract the part-of-speech of the first token after tokenizing the input
        message. Return True if the first token is a verb and False otherwise.
        '''
        # tokenize each sentence into words and tag part of speech
        tokens = tokenize(text)
        tokens = ['nul'] if len(tokens) == 0 else tokens  # avoid empty sentence
        pos_tags = nltk.pos_tag([tokens[0]])

        # index pos_tags to get the first word and part of speech tag
        first_word, first_tag = pos_tags[0]

        # return true if the first word is an appropriate verb or RT for retweet
        if first_tag in ['VB', 'VBP'] or first_word == 'RT':
            return True

        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = [self.starting_verb(x) for x in X]

        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Builds a pipeline for text processing and machine learning classifier.
    Using GridSearchCV for tuning hyperparameters.
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf_pl', Pipeline([
                ('countv', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('startverb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5, verbose=1)))
    ])

    parameters = {
        'features__tfidf_pl__tfidf__norm': ['l2', 'l1'],
        # 'features__tfidf_pl__tfidf__use_idf': [True, False],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names: list[str]):
    '''
    Evaluate the classifier model on the test dataset. Print out to the terminal
    a report for the performance metrics (precision/recall/f1- scores, etc.) for
    each message category using `sklearn.metrics.classification_report`.
    '''
    Y_predict = model.predict(X_test)
    for i, col in enumerate(category_names):
        print('category:', col)
        print(classification_report(Y_test[col].to_numpy(), Y_predict[:, i]))


def save_model(model, model_filepath: str):
    '''
    Exports the trained model as a pickle file.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model (It may take 15 mins)...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
