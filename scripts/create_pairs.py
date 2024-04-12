import os

augmented_train_dataset = "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/augmented_train_dataset/"
unique_person_array = [
    f for f in os.listdir(augmented_train_dataset) if not f.endswith(".DS_Store")
]
# We should have 751 people
assert len(unique_person_array) == 751

features = dict(
    zip(unique_person_array, [([], []) for _ in range(len(unique_person_array))])
)

# populate our features dictionary
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
    assert len(image_features) == len(text_features)
    features[p] = [image_features, text_features]

assert len(features) == 751


# create a dictionary containing the keys are the unique people and the values are
# [[person1_image1_feature, person1_image2_feature...], [person1_text1_feature, person1_text2_feature...]]
