import torch
from PIL import Image
from transformers import TextStreamer
import os
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

# This does the loading of the model. Do not put this into the gen_caption function itself or it will become slow since the model loads everytime it runs.
model_path = "MAGAer13/mplug-owl2-llama2-7b"

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda"
)


# Create the actual function here
def generate_caption(img_path, query):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(img_path).convert("RGB")
    max_edge = max(
        image.size
    )  # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=None,  # put this to None for now because i do not care about seeing a text stream.
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

    return outputs


# Enter the training_directory here which contains the folders of each unique person (751 for this specific dataset)
train_dir = "/home/brytech/vision_question_answer/Market/train/"
unique_person_array = os.listdir(train_dir)

for person in unique_person_array:
    # get the images of the unique person
    specific_person_images = os.listdir(train_dir + person)
    for single_image_of_person in specific_person_images:
        if single_image_of_person.endswith(".txt"):
            continue

        img_path = train_dir + person + "/" + single_image_of_person
        name_of_caption_file_to_save = img_path.split("/")[-1].split(".")[0] + ".txt"

        query = "Describe this person focussing on their clothes and their race. I want the color of their top and bottom, the material of the top and bottom. I want their race and skin color. Do not give me any other information."
        print(f"Working on {name_of_caption_file_to_save}....")
        generated_caption = generate_caption(img_path, query)

        # save the caption to a file
        with open(f"{train_dir + person}/{name_of_caption_file_to_save}", "w") as file:
            file.write(generated_caption)
            print(f"Finished {name_of_caption_file_to_save}!")
