from hstest import StageTest, CheckResult, dynamic_test
import os
import re
import pandas as pd
import pickle


class GoodsPredictions(StageTest):

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_dir, "../data")

    @dynamic_test()
    def check_data(self):
        if not os.path.exists(self.data_path):
            return CheckResult.wrong("There is no directory called data")

        if "preprocessed_df.pkl" not in os.listdir(self.data_path):
            return CheckResult.wrong("The preprocessed_df.pkl file is not in the data directory")

        if "w2v_model.pkl" not in os.listdir(self.data_path):
            return CheckResult.wrong("The w2v_model.pkl file is not in the data directory")

        if "train_df.pkl" not in os.listdir(self.data_path):
            return CheckResult.wrong("The train_df.pkl file is not in the data directory")

        if "test_df.pkl" not in os.listdir(self.data_path):
            return CheckResult.wrong("The test_df.pkl file is not in the data directory")

        if "train_labels.pkl" not in os.listdir(self.data_path):
            return CheckResult.wrong("The train_labels.pkl file is not in the data directory")

        if "test_labels.pkl" not in os.listdir(self.data_path):
            return CheckResult.wrong("The test_labels.pkl file is not in the data directory")

        return CheckResult.correct()

    @dynamic_test()
    def check_w2v_model(self):
        with open(f"{self.data_path}/w2v_model.pkl", "rb") as f:
            model = pickle.load(f)

        if model.vector_size != 300:
            return CheckResult.wrong("The Word2Vec vector_size should be 300")

        if model.epochs != 15:
            return CheckResult.wrong("The Word2Vec epochs should be 15")

        if model.window != 5:
            return CheckResult.wrong("The Word2Vec window should be 5")

        if model.min_count != 5:
            return CheckResult.wrong("The Word2Vec min_count should be 5")

        if model.negative != 20:
            return CheckResult.wrong("The Word2Vec negative should be 20")

        if model.sample != 6e-05:
            return CheckResult.wrong("The Word2Vec sample should be 6e-5")

        if model.sg != 0:
            return CheckResult.wrong("The Word2Vec sg should be 0")

        if model.corpus_count != 40411:
            return CheckResult.wrong("The Word2Vec corpus_count should be 40411")

        if model.corpus_total_words != 2080365:
            return CheckResult.wrong("The Word2Vec corpus_total_words should be 2080365")

        return CheckResult.correct()
