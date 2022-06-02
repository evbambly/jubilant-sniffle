import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import ClassifierI, SklearnClassifier

import random
import pickle
import os
from statistics import mode
from enum import Enum

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = list(classifiers)

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            votes.append(c.classify(features))
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            votes.append(c.classify(features))
        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)


class DatasetTypes(Enum):
    SHUFFLED = "SHUFFLED"
    POS = "POS"
    NEG = "NEG"


class SavedDirs(Enum):
    TRAINED_CLASSIFIERS = "trained_classifiers"
    FEATURE_SETS = "feature_sets"
    SORTED_DOCS = "sorted_docs"
    COMMON_WORDS = "common_words"


def get_training_data():
    reviews_dir = "short_reviews/"
    short_pos_path = "short_pos.txt"
    short_neg_path = "short_neg.txt"
    documents = []

    def add_docs(reviews, doc_type):
        for r in reviews.split('\n'):
            documents.append((word_tokenize(r), doc_type))

    all_words = []
    # J is adjective, r is adverb, v is verb
    allowed_word_types = ["J", "R", "V"]

    # allowed_word_types = ["J"]

    def add_words(file):
        words = word_tokenize(file)
        part_of_speech_tag = nltk.pos_tag(words)
        for tag in part_of_speech_tag:
            word = tag[0]
            pos = tag[1]
            if pos[0] in allowed_word_types and len(word) > 1:
                all_words.append(word.lower())

    with open(reviews_dir + short_neg_path, "r") as short_neg_file, open(reviews_dir + short_pos_path,
                                                                         "r") as short_pos_file:
        short_neg = short_neg_file.read()
        short_pos = short_pos_file.read()
        add_docs(short_neg, "neg")
        add_docs(short_pos, "pos")
        add_words(short_pos)
        add_words(short_neg)

    random.shuffle(documents)
    print("shuffled", len(documents))
    common_words = get_common_words(all_words, 5000, "short_reviews")
    return {"documents": documents, "common_words": common_words}


def get_or_create_data(path, create_data_func):
    full_path = path + ".pickle"
    if os.path.exists(full_path) and os.path.getsize(full_path):
        with open(full_path, "rb") as found_file:
            return pickle.load(found_file)
    else:
        data_to_save = create_data_func()
        save_file = open(full_path, "wb")
        pickle.dump(data_to_save, save_file)
        save_file.close()
        return data_to_save


def get_or_train_model(training_set, classifier, classifier_name, training_set_name):
    classifier_file_path = SavedDirs.TRAINED_CLASSIFIERS.value + "/" + classifier_name + "_" + training_set_name
    return get_or_create_data(classifier_file_path, lambda: classifier.train(training_set))


def find_features(document, most_common_words):
    words = set(document)
    features = {}
    for w in most_common_words:
        features[w] = (w in words)
    return features


def get_feature_sets(documents, common_words, corpus_name):
    feature_set_file_path = SavedDirs.FEATURE_SETS.value + "/" + corpus_name + "_" + str(len(common_words))

    def create_feature_sets():
        feature_sets = []
        for (document, category) in documents:
            next_feature_set = (find_features(document, common_words), category)
            feature_sets.append(next_feature_set)
        return feature_sets

    return get_or_create_data(feature_set_file_path, create_feature_sets)


def get_documents(corpus, shuffle, corpus_name):
    document_file_path = SavedDirs.SORTED_DOCS.value + "/" + corpus_name

    def sort_documents():
        return [(list(corpus.words(fileid)), category)
                for category in corpus.categories()
                for fileid in corpus.fileids(category)]

    documents = get_or_create_data(document_file_path, sort_documents)

    if shuffle:
        random.shuffle(documents)
        print("shuffled", len(documents))
    return documents


def get_common_words(corpus_words, word_limit, corpus_name):
    common_words_file_path = SavedDirs.COMMON_WORDS.value + "/" + corpus_name + "_" + str(word_limit)

    def create_common_words():
        all_words = []
        for w in corpus_words:
            all_words.append(w.lower())
        all_words = nltk.FreqDist(all_words)
        return list(all_words.keys())[:word_limit]

    return get_or_create_data(common_words_file_path, create_common_words)


def get_corpus_data(corpus, corpus_words, corpus_name, shuffle_documents, common_word_count):
    documents = get_documents(corpus, shuffle_documents, corpus_name)
    print("documents loaded", len(documents))
    common_words = get_common_words(corpus_words, common_word_count, corpus_name)
    print("most common words found", common_word_count)
    return {"documents": documents, "common_words": common_words}


def split_training_testing_sets(dataset_type: DatasetTypes, feature_sets):
    testing_set_length = len(feature_sets) // 20

    if dataset_type == DatasetTypes.POS.value:
        training_set = feature_sets[:-testing_set_length]
        testing_set = feature_sets[-testing_set_length:]
    else:
        training_set = feature_sets[testing_set_length:]
        testing_set = feature_sets[:testing_set_length]

    return {"training_set": training_set, "testing_set": testing_set}


def get_learning_datasets(dataset_type, corpus, corpus_words, corpus_name, common_word_count):
    shuffle_documents = dataset_type == DatasetTypes.SHUFFLED.value
    documents, common_words = get_corpus_data(corpus, corpus_words, corpus_name, shuffle_documents,
                                              common_word_count).values()
    feature_sets = get_feature_sets(documents, common_words, corpus_name)
    print("feature sets found")
    return split_training_testing_sets(dataset_type, feature_sets)


def nb_sklearn_classifiers(training_set, testing_set, testing_set_name, should_test=False):
    def get_classifier(classifier, classifier_name):
        return get_or_train_model(training_set, classifier, classifier_name, testing_set_name)

    mnb_classifier = get_classifier(SklearnClassifier(MultinomialNB()), "MultinomialNB")
    print("MultinomialNB Trained")
    if should_test:
        print("mnb classifier", (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)

    bnb_classifier = get_classifier(SklearnClassifier(BernoulliNB()), "BernoulliNB")
    print("BernoulliNB Trained")
    if should_test:
        print("bnb classifier", (nltk.classify.accuracy(bnb_classifier, testing_set)) * 100)

    LogisticRegression_classifier = get_classifier(SklearnClassifier(LogisticRegression()), "LogisticRegression")
    print("LogisticRegression Trained")
    if should_test:
        print("LogisticRegression classifier",
              (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

    SGDClassifier_classifier = get_classifier(SklearnClassifier(SGDClassifier()), "SGDClassifier")
    print("SGDClassifier Trained")
    if should_test:
        print("SGDClassifier classifier", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

    LinearSVC_classifier = get_classifier(SklearnClassifier(LinearSVC()), "LinearSVC")
    print("LinearSVC Trained")
    if should_test:
        print("LinearSVC classifier", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

    return [mnb_classifier, bnb_classifier, LogisticRegression_classifier, SGDClassifier_classifier,
            LinearSVC_classifier ]


def reset_data():
    pass


def sentiment(text):
    documents, common_words = get_training_data().values()
    feature_sets = get_feature_sets(documents, common_words, "short_reviews")
    training_set, testing_set = split_training_testing_sets(DatasetTypes.POS.value, feature_sets).values()
    classifiers = nb_sklearn_classifiers(training_set, testing_set, "short_reviews", True)
    voted_classifier = VoteClassifier(*classifiers)

    features = find_features(text, common_words)
    print(nltk.classify.accuracy(voted_classifier, testing_set))
    return voted_classifier.classify(features), voted_classifier.confidence(features)
