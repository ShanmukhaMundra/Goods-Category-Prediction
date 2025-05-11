def main():
    import pandas as pd
    data = pd.read_csv("data/descriptions.csv")
    data = data.dropna()
    data = data[~(data.description.str.match(r'^[\s]+$|^[#]+$|^[-]+$|^\.$|^[0-9\s\-\.]+$|^\b(?:\w+\s*){1,2}\b$|No\. 30247: 3\/8"-24M'))]
    print(data.head())
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
    print(*preprocessed_data.tokens.tolist()[:4], sep="\n")
    import pickle
    with open("data/preprocessed_df.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f, pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()