import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, wordnet, movie_reviews
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier, ClassifierI

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

import random
import pickle
import os
from statistics import mode
from datetime import  datetime

import sentiment_mod as s


def tokenize():
    example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is " \
                   "pinkish-blue. You should not eat cardboard "

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(example_text)

    filled_sentence = [w for w in words if w not in stop_words]

    ps = PorterStemmer()

    example_words = ["python", "pythoner", "pythonning", "pythoned", "pythonly", "path"]

    stemmed_words = set([ps.stem(w) for w in example_words])

    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")

    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

    return (custom_sent_tokenizer.tokenize(sample_text), stop_words)


def process_content():
    (tokenized, stop_words) = tokenize()
    try:
        sotu_word_tokenized = [nltk.word_tokenize(w) for w in tokenized]
        sotu_stopped = []
        for token in sotu_word_tokenized:
            sotu_stopped.append([w for w in token if w not in stop_words])
        sotu_tagged = map(lambda t: nltk.pos_tag(t), sotu_stopped)
        print(sotu_word_tokenized)
        print(sotu_stopped)
        print(list(sotu_tagged))

    except Exception as e:
        print(str(e))


# Lesson 5

def chunked_content():
    tokenized = tokenize()[0]
    try:
        for i in tokenized[6:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<RB.?>*<NNP>+<NN>?} 
            }<VB.?|IN|DT>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            namedEnt = nltk.ne_chunk(tagged)
            namedEntUnspecified = nltk.ne_chunk(tagged, binary=True)
            print(namedEntUnspecified)
            print(namedEntUnspecified[:1])



    except Exception as e:
        print(str(e))


# Lesson 8

lemmatizer = WordNetLemmatizer()


def lemmatize_example():
    print(lemmatizer.lemmatize("cats"))
    print(lemmatizer.lemmatize("cacti"))
    print(lemmatizer.lemmatize("python"))
    print(lemmatizer.lemmatize("better", pos="a"))
    print(lemmatizer.lemmatize("better"))


# print(nltk.__file__)

# Lesson 10


def get_synonnyms():
    synonyms = wordnet.synsets("program")
    print(synonyms[0].lemmas())
    print(synonyms[0].lemmas()[0].name())
    print(synonyms[0].definition())
    print(synonyms[0].examples())


def get_antonyms():
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    print(set(synonyms))
    print(set(antonyms))


def get_similarities():
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("boat.n.01")
    print(w1.wup_similarity(w2))

    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("car.n.01")
    print(w1.wup_similarity(w2))

    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("cat.n.01")
    print(w1.wup_similarity(w2))


# Lesson 11

def get_most_common_words(corpus_words, word_limit):
    all_words = []
    for w in corpus_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    return list(all_words.keys())[:word_limit]


def find_features(document, most_common_words):
    words = set(document)
    features = {}
    for w in most_common_words:
        features[w] = (w in words)
    return features


def get_documents(should_shuffle):
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    if should_shuffle:
        random.shuffle(documents)
        print("shuffled", len(documents))
    return documents


def text_classification():
    # print(all_words.most_common(10))
    # print(all_words["smart"])

    feat1 = find_features(movie_reviews.words("neg/cv000_29416.txt"),
                          get_most_common_words(movie_reviews.words(), 3000))
    true_feat1 = [f for f in feat1 if next(iter(feat1[f]))]
    false_feat1 = [f for f in feat1 if not next(iter(feat1[f]))]
    # print(true_feat1)
    # print(false_feat1)


# Lesson 13
dataset_types = dict.fromkeys(["SHUFFLED", "POS", "NEG"])


def get_learning_datasets(dataset_type, documents, common_words):
    feature_sets = [(find_features(rev, common_words), category) for (rev, category) in documents]
    print("featured", len(feature_sets))

    testing_set_length = len(feature_sets) // 20

    if dataset_type == "POS":
        training_set = feature_sets[:-testing_set_length]
        testing_set = feature_sets[-testing_set_length:]
    else:
        training_set = feature_sets[testing_set_length:]
        testing_set = feature_sets[:testing_set_length]

    return {"training_set": training_set, "testing_set": testing_set}


def get_or_train_model(training_set, classifier, classifier_name):
    classifier_file_path = "trained_classifiers/" + classifier_name + ".pickle"
    if os.path.exists(classifier_file_path) and os.path.getsize(classifier_file_path):
        print("found ", classifier_name)
        with open(classifier_file_path, "rb") as classifier_file:
            return pickle.load(classifier_file)
    else:
        trained_classifier = classifier.train(training_set)
        save_classifier = open(classifier_file_path, "wb")
        pickle.dump(trained_classifier, save_classifier)
        save_classifier.close()
        print("trained ", classifier_name)
        return trained_classifier

def get_movie_reviews_datasets():
    training_set, testing_set = get_learning_datasets("SHUFFLED", get_documents(True),
                                                      get_most_common_words(movie_reviews.words(), 3000)).values()
    return {"training_set": training_set, "testing_set": testing_set}

def naive_bayes(training_set, testing_set):
    classifier = get_or_train_model(training_set, nltk.NaiveBayesClassifier, "naivebayes")

    print("NB Accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    classifier.show_most_informative_features(15)


# Lesson 15

def nb_sklearn_classifiers(training_set, testing_set, should_test=False):
    mnb_classifier = get_or_train_model(training_set, SklearnClassifier(MultinomialNB()), "MultinomialNB")
    if should_test:
        print("mnb classifier", (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)

    # gnb_classifier = SklearnClassifier(GaussianNB())
    # gnb_classifier.train(training_set)
    # print("gnb classifier", (nltk.classify.accuracy(gnb_classifier, testing_set)) * 100)

    bnb_classifier = get_or_train_model(training_set, SklearnClassifier(BernoulliNB()), "BernoulliNB")
    if should_test:
        print("bnb classifier", (nltk.classify.accuracy(bnb_classifier, testing_set)) * 100)

    LogisticRegression_classifier = get_or_train_model(training_set, SklearnClassifier(LogisticRegression()),
                                                       "LogisticRegression")
    if should_test:
        print("LogisticRegression classifier",
              (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

    SGDClassifier_classifier = get_or_train_model(training_set, SklearnClassifier(SGDClassifier()), "SGDClassifier")
    if should_test:
        print("SGDClassifier classifier", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

    LinearSVC_classifier = get_or_train_model(training_set, SklearnClassifier(LinearSVC()), "LinearSVC")
    if should_test:
        print("LinearSVC classifier", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

    NuSVC_classifier = get_or_train_model(training_set, SklearnClassifier(NuSVC()), "NuSVC")
    if should_test:
        print("NuSVC classifier", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

    return [mnb_classifier, bnb_classifier, LogisticRegression_classifier, SGDClassifier_classifier,
            LinearSVC_classifier,
            NuSVC_classifier]


# Lesson 16

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


def combine_algos(training_set, testing_set):
    classifiers = nb_sklearn_classifiers(training_set, testing_set)
    voted_classifier = VoteClassifier(*classifiers)

    print("voted classifier", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
    print("Classification:", voted_classifier.classify(testing_set[0][0]),
          "Confidence:", voted_classifier.confidence(testing_set[0][0]))
    print("Classification:", voted_classifier.classify(testing_set[1][0]),
          "Confidence:", voted_classifier.confidence(testing_set[1][0]))
    print("Classification:", voted_classifier.classify(testing_set[2][0]),
          "Confidence:", voted_classifier.confidence(testing_set[2][0]))
    print("Classification:", voted_classifier.classify(testing_set[3][0]),
          "Confidence:", voted_classifier.confidence(testing_set[3][0]))
    print("Classification:", voted_classifier.classify(testing_set[4][0]),
          "Confidence:", voted_classifier.confidence(testing_set[4][0]))


# Lesson 17


def investigating_bias():
    start_time = datetime.now()
    documents = get_documents(False)
    common_words = get_most_common_words(movie_reviews.words(), 3000)
    data_gathering_done = datetime.now()
    print("data gathering done ", (data_gathering_done - start_time).total_seconds())

    training_set, testing_set = get_learning_datasets("POS", documents, common_words).values()
    print("POS")
    combine_algos(training_set, testing_set)
    pos_done = datetime.now()
    print("pos done ", (pos_done - start_time).total_seconds())

    training_set, testing_set = get_learning_datasets("NEG", documents, common_words).values()
    print("NEG")
    combine_algos(training_set, testing_set)
    neg_done = datetime.now()
    print("neg done ", (neg_done - start_time).total_seconds())

    documents = get_documents(True)
    training_set, testing_set = get_learning_datasets("SHUFFLED", documents, common_words).values()
    print("SHUFFLED")
    combine_algos(training_set, testing_set)
    shuffle_done = datetime.now()
    print("shuffle done ", (shuffle_done - start_time).total_seconds())


# Lesson 18

def get_training_data():
    reviews_dir = "short_reviews/"
    short_pos_path = "short_pos.txt"
    short_neg_path = "short_neg.txt"
    documents = []

    def add_docs(reviews, type):
        for r in reviews.split('\n'):
            documents.append((r, type))

    all_words = []

    def add_words(words):
        for w in words:
            all_words.append(w.lower())

    with open(reviews_dir + short_neg_path, "r") as short_neg_file, open(reviews_dir + short_pos_path,
                                                                         "r") as short_pos_file:
        short_neg = short_neg_file.read()
        short_pos = short_pos_file.read()
        add_docs(short_neg, "neg")
        add_docs(short_pos, "pos")
        short_pos_words = word_tokenize(short_pos)
        short_neg_words = word_tokenize(short_pos)
        add_words(short_pos_words)
        add_words(short_neg_words)

    common_words = get_most_common_words(all_words, 5000)
    return {"documents": documents, "common_words": common_words}


def using_better_training_data():
    documents, common_words = get_training_data().values()
    training_set, testing_set = get_learning_datasets("SHUFFLED", documents, common_words).values()

    #naive_bayes(training_set, testing_set)
    #combine_algos(training_set, testing_set)
    investigating_bias()



def main():
    print(s.sentiment("This was a great movie! Enjoyed every minute of it! Brilliant execution. Good positive pos genius"), "pos")
    print(s.sentiment("This movie was utter junk. What's the point?"), "neg")


if __name__ == "__main__":
    main()