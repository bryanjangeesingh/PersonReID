import os
import pickle

train_super_features_dir = "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/super_feature_dataset/"
unique_person_array = [
    f for f in os.listdir(train_super_features_dir) if not f.endswith(".DS_Store")
]

# first create a features dictionary
features = dict(zip(unique_person_array, [[] for _ in range(len(unique_person_array))]))

for person in unique_person_array:
    for super_feature in os.listdir(train_super_features_dir + "/" + person):
        features[person].append(super_feature)

assert len(features) == 751


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# save this feature dictionary as a pkl
with open(
    "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/super_feature_dataset/features_dict.pkl",
    "wb",
) as f:
    pickle.dump(features, f)
