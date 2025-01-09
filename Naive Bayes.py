import string
from collections import defaultdict

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

def build_vocabulary(dataset):
    vocabulary = set()
    for text, _ in dataset:
        words = preprocess_text(text)
        vocabulary.update(words)
    return list(vocabulary)

def calculate_probabilities(dataset, vocabulary):
    class_counts = defaultdict(int)
    word_counts = {0: defaultdict(int), 1: defaultdict(int)}
    total_words = {0: 0, 1: 0}

    for text, label in dataset:
        class_counts[label] += 1
        words = preprocess_text(text)
        for word in words:
            if word in vocabulary:
                word_counts[label][word] += 1
                total_words[label] += 1

    total_documents = len(dataset)
    prior_probs = {
        0: class_counts[0] / total_documents,
        1: class_counts[1] / total_documents,
    }

    word_likelihoods = {0: {}, 1: {}}
    for word in vocabulary:
        word_likelihoods[0][word] = (word_counts[0][word] + 1) / (total_words[0] + len(vocabulary))
        word_likelihoods[1][word] = (word_counts[1][word] + 1) / (total_words[1] + len(vocabulary))

    return prior_probs, word_likelihoods

def naive_bayes_classifier(text, prior_probs, word_likelihoods, vocabulary):
    words = preprocess_text(text)
    scores = {0: prior_probs[0], 1: prior_probs[1]}

    for label in [0, 1]:
        for word in words:
            if word in vocabulary:
                scores[label] *= word_likelihoods[label][word]

    return 1 if scores[1] > scores[0] else 0

def evaluate_classifier(test_data, prior_probs, word_likelihoods, vocabulary):
    correct = 0
    for text, label in test_data:
        prediction = naive_bayes_classifier(text, prior_probs, word_likelihoods, vocabulary)
        if prediction == label:
            correct += 1
    return correct / len(test_data)

train_data = [
    ("I love this movie", 1),
    ("This film is fantastic", 1),
    ("What an amazing experience", 1),
    ("I dislike this movie", 0),
    ("Not a great film", 0),
    ("This is terrible", 0),
]

test_data = [
    ("I love this film", 1),
    ("This movie is terrible", 0),
]

vocabulary = build_vocabulary(train_data)
prior_probs, word_likelihoods = calculate_probabilities(train_data, vocabulary)
accuracy = evaluate_classifier(test_data, prior_probs, word_likelihoods, vocabulary)

print(f"Vocabulary: {vocabulary}")
print(f"Prior Probabilities: {prior_probs}")
print(f"Accuracy: {accuracy}")
