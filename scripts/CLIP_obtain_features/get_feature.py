import torch
import clip
from PIL import Image
import os
import pickle

# change this to cuda if we decide to run this on Satori

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)


def save_tensor_as_pkl(tensor, save_directory, filename):
    """
    Helper function to save a tensor as a pickle file.
    Parameters:
        tensor (torch.Tensor): The input tensor to be saved.
        save_directory (str): The directory where the tensor will be saved.
        filename (str): The name of the pickle file.
    """

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, filename)

    with open(save_path, "wb") as f:
        pickle.dump(tensor.cpu(), f)
        print(f"Finished savint to {save_path}!")


# load the model outside of the function to prevent having to load it multiple times
model, preprocess = clip.load("ViT-B/32", device=device)


def get_text_and_image_features_from_clip(image_path: str, text_path: str) -> tuple:
    """
    Returns the image and text features from the CLIP model given an image path and a text path.

    Parameters:
        image_path (str): The path to the image file.
        text_path (str): The path to the text file.

    Returns:
        tuple: A tuple containing the image and text features as tensors.
    """

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with open(text_path, "r") as file:
        text_input = file.read()

    assert type(text_input) == str

    text = clip.tokenize([text_input]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    return image_features, text_features


# Enter the training_directory here which contains the folders of each unique person (751 for this specific dataset)
train_dir = "/home/brytech/vision_question_answer/Market/train/"
augmented_dataset_dir = ""
unique_person_array = os.listdir(train_dir)

for person in unique_person_array:
    # get the images of the unique person
    specific_person_images = os.listdir(train_dir + person)
    # create a folder in the persons' folder for the feature repr. of the images and text
    os.makedirs(augmented_dataset_dir + person + "/features", exist_ok=True)

    for single_image_of_person in specific_person_images:
        img_path = train_dir + person + "/" + single_image_of_person + ".jpg"
        txt_path = train_dir + person + "/" + single_image_of_person + ".txt"

        base_name_to_save = img_path.split("/")[-1].split(".")[0]

        # append either img or txt at the end of base_name_to_save depending on what we are saving
        image_features, text_features = get_text_and_image_features_from_clip(
            img_path, txt_path
        )

        # save these off into pkls for later use
        save_tensor_as_pkl(
            image_features,
            augmented_dataset_dir + person + "/features",
            base_name_to_save + "_image_features.pkl",
        )
        save_tensor_as_pkl(
            text_features,
            augmented_dataset_dir + person + "/features",
            base_name_to_save + "_text_features.pkl",
        )
