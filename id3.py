import pandas as pd
import numpy as np



def calculate_entropy(data, target_col):
    values, counts = np.unique(data[target_col], return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / sum(counts)) * calculate_entropy(data[data[attribute] == values[i]], target_col)
        for i in range(len(values))
    ])
    info_gain = total_entropy - weighted_entropy
    return info_gain

def build_tree(data, attributes, target_col, parent_class=None):
    if len(np.unique(data[target_col])) == 1:
        return np.unique(data[target_col])[0]
    
    if len(attributes) == 0:
        return parent_class
    
    parent_class = data[target_col].mode()[0]

    info_gains = {attr: calculate_information_gain(data, attr, target_col) for attr in attributes}
    best_attribute = max(info_gains, key=info_gains.get)
    

    tree = {best_attribute: {}}
    attributes = [attr for attr in attributes if attr != best_attribute]

    for value in np.unique(data[best_attribute]):
        subset = data[data[best_attribute] == value]
        subtree = build_tree(subset, attributes, target_col, parent_class)
        tree[best_attribute][value] = subtree
    
    return tree


def predict(tree, data_point):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = data_point.get(attribute)
    if value not in tree[attribute]:
        return None 
    return predict(tree[attribute][value], data_point)

if __name__ == "__main__":
    data = pd.DataFrame({
        "Weather": ["Sunny", "Overcast", "Rainy", "Sunny", "Sunny", "Overcast", "Rainy"],
        "Temperature": ["Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild"],
        "Play?": ["No", "Yes", "Yes", "Yes", "No", "Yes", "Yes"]
    })
    
    target_col = "Play?"
    attributes = [col for col in data.columns if col != target_col]
    
    decision_tree = build_tree(data, attributes, target_col)
    print("Decision Tree:", decision_tree)

    test_data = {"Weather": "Sunny", "Temperature": "Cool"}
    prediction = predict(decision_tree, test_data)
    print("Prediction for", test_data, ":", prediction)
