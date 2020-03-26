# Train, test, and cache the basic classifier for Obfuscation HW
import pandas
import sklearn
import argparse
from sklearn.metrics import accuracy_score
from nltk import tokenize
import pickle

def get_preds(cache_name, test):
    m,v = pickle.load(open(cache_name, 'rb'))
    test = [" ".join(tokenize.word_tokenize(t)) for t in test]
    test_data_features = v.transform(test)
    preds = m.predict(test_data_features)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="./test.csv")
    args = parser.parse_args()

    test_data = pandas.read_csv(args.test_file)

    cache_name = 'gender_classifier.pickle'
    test_preds = get_preds(cache_name, list(test_data["post_text"]))
    gold_test = list(test_data["op_gender"])

    print("Gender classification accuracy", accuracy_score(list(test_preds), gold_test))

    cache_name = 'subreddit_classifier.pickle'
    test_preds = get_preds(cache_name, list(test_data["post_text"]))
    gold_test = list(test_data["subreddit"])
    print("Subreddit accuracy", accuracy_score(list(test_preds), gold_test))

if __name__ == "__main__":
    main()
