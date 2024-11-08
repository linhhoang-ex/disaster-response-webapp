import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    '''
    Load data from csv files and merge the two datassets to a pandas dataframe.

    Params
    ------
    - `messages_filepath`: filepath to the csv file for disaster messages
    - `categories_filepath`: filepath to the csv file for message categories

    Return
    ------
    `df`: a dataframe merging categories and messages data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the merged dataset in the form of a dataframe and return the cleaned
    data as a dataframe.
    '''
    # Split the 'categories' column into multiple seperate category columns
    categories = df['categories'].str.split(";").apply(pd.Series)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).to_list()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        categories.loc[categories[column] > 1, column] = 1

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df: pd.DataFrame, database_filename='DisasterPipelineDB.db'):
    '''Export the cleaned dataset (as a dataframe) into an sqlite database.'''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessage', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
