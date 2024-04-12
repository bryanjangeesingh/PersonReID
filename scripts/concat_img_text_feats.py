import os
import pickle
import numpy as np

augmented_train_dataset = "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/augmented_train_dataset/"
unique_person_array = [
    f for f in os.listdir(augmented_train_dataset) if not f.endswith(".DS_Store")
]

# store the new super feature dataset here
train_super_feature_dataset = "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/super_feature_dataset/"
# if the above directory doesn't exist, create it
if not os.path.exists(train_super_feature_dataset):
    os.makedirs(train_super_feature_dataset)

# We should have 751 people
assert len(unique_person_array) == 751


# write a function to accept the path of a pkl file and load it back into a tensor
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


count = 0
for p in unique_person_array:
    persons_content = [
        f
        for f in os.listdir(augmented_train_dataset + p + "/" + "features/")
        if not f.endswith(".DS_Store")
    ]

    image_features = list(
        filter(lambda x: x.endswith("_image_features.pkl"), persons_content)
    )

    text_features = list(
        filter(lambda x: x.endswith("_text_features.pkl"), persons_content)
    )

    # sort the image and text features so they are in the same order
    image_features.sort()
    text_features.sort()
    count += 1
    for i in range(len(image_features)):
        # concat the text and image features into a single tensor
        image = load_pkl(
            augmented_train_dataset + p + "/" + "features/" + image_features[i]
        )
        text = load_pkl(
            augmented_train_dataset + p + "/" + "features/" + text_features[i]
        )

        super_feature = np.concatenate((image, text), axis=1)
        # save the super feature to a pkl in a new directory that is a folder containing folders of the peoples name followed by all their super features

        # create the person's folder if it doesn't exist
        if not os.path.exists(train_super_feature_dataset + p):
            os.makedirs(train_super_feature_dataset + p)

        # save the super feature
        with open(
            train_super_feature_dataset
            + p
            + "/"
            + f'{image_features[i].split(".")[0]}_super_feature.pkl',
            "wb",
        ) as f:
            pickle.dump(super_feature, f)
            print(
                f'Finished saving {train_super_feature_dataset + p + "/" + "super_feature"}!'
            )
            print(f"Progress {((count)/len(unique_person_array))*100}%")
