def main():
    import pandas as pd
    data = pd.read_csv("data/descriptions.csv")
    data = data.dropna()
    data = data[~(data.description.str.match(r'^[\s]+$|^[#]+$|^[-]+$|^\.$|^[0-9\s\-\.]+$|^\b(?:\w+\s*){1,2}\b$|No\. 30247: 3\/8"-24M'))]
    #print(data.head())
    import re
    from nltk import regexp_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    def preprocess_data(data):
        def tokenize_text(text):
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

        def lemmatize_text(tokens):
            lemmatizer = WordNetLemmatizer()
            tokens = pos_tag(tokens)
            tokens = [lemmatizer.lemmatize(token, pos=f"{tag_map(tag)}") for token, tag in tokens]
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
        data["tokens"] = data["description"].apply(lambda x: lemmatize_text(tokenize_text(x)))
        return data

    preprocessed_data = preprocess_data(data)
    #print(*preprocessed_data.tokens.tolist()[:4], sep="\n")
    import pickle
    with open("data/preprocessed_df.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f, pickle.HIGHEST_PROTOCOL)

    #Loading "preprocessed_df.pkl" from data directory
    with open("data/preprocessed_df.pkl", "rb") as f:
        df = pickle.load(f)

    #Splittng Data into Train and Test sets
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.1, random_state=10, stratify=df.categories)

    #categories for the train and test sets
    train_labels = train.categories
    test_labels = test.categories

    #preprocessed tokens from train and test sets
    train_tokens = train.tokens
    test_tokens = test.tokens

    from gensim.models import Word2Vec, KeyedVectors
    def train_word2vec(train_data: pd.Series(list[str]), test_data: pd.Series(list[str])) -> \
            [KeyedVectors, pd.DataFrame, pd.DataFrame]:

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
        import numpy as np
        train_df = pd.DataFrame([np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                                         or [np.zeros(300)], axis=0) for tokens in train_data])

        """Generating the test word embeddings vector and save as a DataFrame 
    and finding mean of the embeddings for each word. For words not in the vocabulary
    setting the embeddings as a numpy array of zeros"""
        test_df = pd.DataFrame([np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv] or
                                        [np.zeros(300)], axis=0) for tokens in test_data])

        return w2v_model, train_df, test_df

    w2v_model, train_df, test_df = train_word2vec(train_tokens, test_tokens)
    print(sorted(w2v_model.wv.key_to_index.keys())[:50])

    #saving pickle files to the data directory
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
if __name__ == "__main__":
    main()