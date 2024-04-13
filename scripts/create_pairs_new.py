import os
import pickle
import random


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# create pairs from the features dictionary
def create_pairs(features, base_path):
    positive_pairs = []
    negative_pairs = []
    unique_people = list(features.keys())

    # Create positive pairs
    for person, files in features.items():
        if len(files) > 1:
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    feat1 = load_pkl(os.path.join(base_path, person, files[i]))
                    feat2 = load_pkl(os.path.join(base_path, person, files[j]))
                    positive_pairs.append((feat1, feat2))

    # Create negative pairs
    for person, files in features.items():
        other_people = [p for p in unique_people if p != person]
        for file in files:
            feat1 = load_pkl(os.path.join(base_path, person, file))
            other_person = random.choice(other_people)
            other_file = random.choice(features[other_person])
            feat2 = load_pkl(os.path.join(base_path, other_person, other_file))
            negative_pairs.append((feat1, feat2))

    return positive_pairs, negative_pairs
