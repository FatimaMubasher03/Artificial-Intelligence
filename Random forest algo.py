import numpy as np
import pandas as pd
from collections import Counter
import random

def calculate_entropy(data, target_col):
    target_values = data[target_col]
    counts = Counter(target_values)
    total = len(target_values)
    entropy = -sum((count / total) * np.log2(count / total) for count in counts.values())
    return entropy

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_col)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

def build_tree(data, attributes, target_col, depth=0, max_depth=3):
    if depth == max_depth or len(attributes) == 0 or len(data[target_col].unique()) == 1:
        return Counter(data[target_col]).most_common(1)[0][0]
    best_attr = max(attributes, key=lambda attr: calculate_information_gain(data, attr, target_col))
    tree = {best_attr: {}}
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = Counter(data[target_col]).most_common(1)[0][0]
        else:
            tree[best_attr][value] = build_tree(subset, [attr for attr in attributes if attr != best_attr], target_col, depth + 1, max_depth)
    return tree

def predict(tree, data_point):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = data_point.get(attr)
    subtree = tree[attr].get(value, None)
    if subtree is None:
        return None
    return predict(subtree, data_point)

def build_random_forest(data, attributes, target_col, n_trees=2):
    trees = []
    for _ in range(n_trees):
        subset = data.sample(frac=0.8, replace=True) 
        selected_attrs = random.sample(attributes, k=len(attributes) // 2)  
        tree = build_tree(subset, selected_attrs, target_col)
        trees.append(tree)
    return trees

def predict_forest(forest, data_point):
    predictions = [predict(tree, data_point) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

data = pd.DataFrame({
    "Weather": ["Sunny", "Overcast", "Rainy", "Sunny", "Overcast", "Rainy"],
    "Temperature": ["Hot", "Hot", "Mild", "Mild", "Mild", "Hot"],
    "Play?": ["No", "Yes", "Yes", "No", "Yes", "No"]
})

attributes = ["Weather", "Temperature"]
target_col = "Play?"

forest = build_random_forest(data, attributes, target_col, n_trees=2)

test_data_point = {"Weather": "Sunny", "Temperature": "Mild"}
prediction = predict_forest(forest, test_data_point)
print("Prediction:", prediction)
