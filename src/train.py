import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import sys

stemmer = PorterStemmer()

def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence.lower())

def stem_sentence(sentence):
    words = sentence.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_sentence(sentence):
    return stem_sentence(remove_punctuation(sentence))

def load_data(path_to_train_data):
    df = pd.read_csv(path_to_train_data, header=None)
    df.columns = ['tweet_id', 'entity', 'sentiment', 'content']
    df = df.dropna()
    df['sentiment'] = pd.Categorical(df['sentiment'])
    df['processed_content'] = df['content'].apply(preprocess_sentence)
    return df

def get_random_forest_pipeline():
    return Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RandomForestClassifier()),
        ]
    )

def fit_and_score_pipeline(df, pipeline):
    train, test = train_test_split(df)
    pipeline.fit(train['processed_content'], train['sentiment'])
    return pipeline.score(test['processed_content'], test['sentiment'])

def save_model(model, path):
    serialized_model = pickle.dumps(model)
    with open(path, 'wb+') as file:
        file.write(serialized_model)


if __name__ == '__main__':
    path_to_train_data, path_to_model = sys.argv[1], sys.argv[2]
    df = load_data(path_to_train_data)
    pipeline_rf = get_random_forest_pipeline()
    accuracy = fit_and_score_pipeline(df, pipeline_rf)
    print(f'On training data accuracy: {accuracy}')
    save_model(pipeline_rf, path_to_model)
    