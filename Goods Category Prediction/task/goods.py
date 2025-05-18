import pandas as pd
import numpy as np
from nltk import regexp_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
import re
from nltk import pos_tag
import streamlit as st
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def tokenize_text(text):
    text = text.lower().strip()
    tokens = re.findall(r'[A-z]+', text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def tag_map(postag):
    if postag.startswith("J"):
        return wordnet.ADJ
    elif postag.startswith("V"):
        return wordnet.VERB
    elif postag.startswith(("R", "D", "P", "X")):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_data(data):
    def tokenize_text_preprocess(text):
        text = (text.lower()
                .replace("]", "")
                .replace("[", "")
                .replace("\\n", "")
                .replace("\\r\\n", "")
                .replace("^", "")
                .replace("_", "")
                .replace("`", "")
                .replace("\\", "").strip()
                )
        #text = re.sub(r"\s+", " ", text)
        tokens = regexp_tokenize(text, pattern='[A-z]+')
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def lemmatize_text_preprocess(tokens):
        lemmatizer = WordNetLemmatizer()
        tokens = pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, pos=f"{tag_map(tag)}") for token, tag in tokens]
        return tokens

    data["tokens"] = data["description"].apply(lambda x: lemmatize_text_preprocess(tokenize_text_preprocess(x)))
    return data


def train_word2vec(train_data: pd.Series, test_data: pd.Series) -> (KeyedVectors, pd.DataFrame, pd.DataFrame):
    #instantiating the Word2Vec with parameters
    w2v_model = Word2Vec(min_count=5, window=5, sg=0, vector_size=300, sample=6e-5, negative=20)

    #Building the vocabulary of w2v_model with the training data
    w2v_model.build_vocab(train_data)

    """Training the w2v_model with the train_data,
and the total_examples set to model corpus count and epochs parameters set to 15"""
    w2v_model.train(train_data, total_examples=w2v_model.corpus_count, epochs=15)

    """Generating the train word embeddings vector and save as a DataFrame 
and finding mean of the embeddings for each word. For words not in the vocabulary
setting the embeddings as a numpy array of zeros"""
    train_df = pd.DataFrame([np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                                     or [np.zeros(300)], axis=0) for tokens in train_data])

    """Generating the test word embeddings vector and save as a DataFrame 
and finding mean of the embeddings for each word. For words not in the vocabulary
setting the embeddings as a numpy array of zeros"""
    test_df = pd.DataFrame([np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv] or
                                    [np.zeros(300)], axis=0) for tokens in test_data])

    return w2v_model, train_df, test_df


def train_rf(train_df, train_labels):
    rf_model = RandomForestClassifier()
    rf_model.fit(train_df, train_labels)
    return rf_model


def load_models():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #Loading w2v_model.pkl from the data directory
    with open(os.path.join(current_dir, "data/w2v_model.pkl"), "rb") as f:
        w2v_model = pickle.load(f)

    # Loading rf_model.pkl from the data directory
    with open(os.path.join(current_dir, "data/rf_model.pkl"), "rb") as f:
       rf_model = pickle.load(f)

    #Loading label_encode.pkl from data directory
    with open(os.path.join(current_dir, "data/label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return w2v_model, rf_model, label_encoder


def predict_top3_categories(text_input, w2v_model, rf_model, label_encoder):
    # Tokenizing & Lemmatizing text input
    tokens = tokenize_text(text_input)
    input_tokens = lemmatize_text(tokens)

    # Getting mean of embedding vector
    input_embed = np.mean([w2v_model.wv[word] for word in input_tokens if word in w2v_model.wv]
                          or [np.zeros(300)], axis=0)

    # converting embedding vectors to a DataFrame
    input_df = pd.DataFrame([input_embed])

    # predicting probabilities of the rf_model with predict_prob
    predicted_proba = rf_model.predict_proba(input_df)

    # getting top3 category indices from the probability values
    top3_idx = np.argsort(predicted_proba, axis=1)[:, -3:]

    # getting predicted value classes and flatten
    top3_classes_encoded = rf_model.classes_[top3_idx]
    top3_classes_encoded_flat = top3_classes_encoded.flatten()

    # performing inverse transformation to get the name of the classes and reshape
    top3_classes = label_encoder.inverse_transform(top3_classes_encoded_flat)
    top3_classes = top3_classes.reshape(top3_classes_encoded.shape)

    return top3_classes, predicted_proba, top3_idx


def process_data():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(current_dir, "data/descriptions.csv"))
    data = data.dropna()
    data = data[~(data.description.str.match(r'^[\s]+$|^[#]+$|^[-]+$|^\.$|^[0-9\s\-\.]+$|^\b(?:\w+\s*){1,2}\b$|No\. 30247: 3\/8"-24M'))]

    preprocessed_data = preprocess_data(data)

    with open("data/preprocessed_df.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f, pickle.HIGHEST_PROTOCOL)

    with open("data/preprocessed_df.pkl", "rb") as f:
        df = pickle.load(f)

    train, test = train_test_split(df, test_size=0.1, random_state=10, stratify=df.categories)

    train_labels = train.categories
    test_labels = test.categories

    train_tokens = train.tokens
    test_tokens = test.tokens

    w2v_model, train_df, test_df = train_word2vec(train_tokens, test_tokens)

    with open("data/w2v_model.pkl", "wb") as f:
        pickle.dump(w2v_model, f, pickle.HIGHEST_PROTOCOL)

    with open("data/train_df.pkl", "wb") as f:
        pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)

    with open("data/test_df.pkl", "wb") as f:
        pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)

    with open("data/train_labels.pkl", "wb") as f:
        pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)

    with open("data/test_labels.pkl", "wb") as f:
        pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)

    train_df = pd.read_pickle("data/train_df.pkl")
    test_df = pd.read_pickle("data/test_df.pkl")

    with open("data/train_labels.pkl", "rb") as f:
        y_train = pickle.load(f)

    with open("data/test_labels.pkl", "rb") as f:
        y_test = pickle.load(f)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(y_train)
    test_labels = label_encoder.transform(y_test)

    rf_model = train_rf(train_df, train_labels)

    predict_test = rf_model.predict(test_df)
    predict_train = rf_model.predict(train_df)

    with open('data/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f, pickle.HIGHEST_PROTOCOL)

    with open("data/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f, pickle.HIGHEST_PROTOCOL)

def main():
    st.title('Goods Category App')

    text_input = st.text_area('Enter Goods Description', 'Type Here...')

    button, space, clear = st.columns([1, 2, 1])

    with button:
        if st.button('Predict'):
            w2v_model, rf_model, label_encoder = load_models()
            top3_classes, predicted_proba, top3_idx = predict_top3_categories(text_input, w2v_model, rf_model,
                                                                              label_encoder)

            with space:
                st.write("Top 3 Predicted Categories:")
                for i, classes in enumerate(top3_classes):
                    for category, prob in zip(classes[::-1], predicted_proba[i, top3_idx[i]][::-1]):
                        st.write(f"{category} : {prob * 100:.0f}")

    with clear:
        clear_button = st.button('Clear')

        if clear_button:
            st.empty()

if __name__ == "__main__":
    main()
