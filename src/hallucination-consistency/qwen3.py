import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import json

logging.set_verbosity_error()

def generate(directory, file_index, prompt, tokenizer, model, enable_thinking, do_sample, temperature):
    # save the output to a file
    os.makedirs(f"../../logs/{directory}/{model_name[5:]}/thinking_{enable_thinking}", exist_ok=True)
    # check if the file exists
    if os.path.exists(f"../../logs/{directory}/{model_name[5:]}/thinking_{enable_thinking}/qa_{file_index}.txt"):
        print(f"Error: file already exists; file_index: {file_index}")
        return

    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=do_sample,
        temperature=temperature,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    with open(f"../../logs/{directory}/{model_name[5:]}/thinking_{enable_thinking}/qa_{file_index}.txt", "w") as f:
        # write to the file
        f.write("**************************************** Inputs ****************************************\n")
        f.write("model: " + model_name + "\n")
        f.write("enable_thinking: " + str(enable_thinking) + "\n")
        f.write("do_sample: " + str(do_sample) + "\n")
        if do_sample:
            f.write("temperature: " + str(temperature) + "\n")
        f.write("prompt:\n" + prompt + "\n")
        f.write("*********************************** End of Inputs **************************************\n\n\n\n\n")

        if enable_thinking:
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            f.write("*************************************** Thinking ***************************************\n")
            f.write(thinking_content + "\n")
            f.write("*********************************** End of Thinking ************************************\n\n\n\n\n")

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        f.write("**************************************** Content ***************************************\n")
        f.write(content + "\n")
        f.write("************************************ End of Content ************************************\n")


if __name__ == "__main__":
    model_size = sys.argv[1]
    if model_size not in ["14B", "32B"]:
        raise ValueError("Model size must be either 14B or 32B")
    model_name = "Qwen/Qwen3-" + model_size
    enable_thinking = sys.argv[2] == "True"
    starting_index = int(sys.argv[3])
    max_index = int(sys.argv[4])
    do_sample = False
    temperature = 0.01
    if len(sys.argv) > 5:
        do_sample = True
        temperature = float(sys.argv[5])

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    ### us places
    with open("../../data/hallucination-consistency/us_places.json", "r") as f:
        places = json.load(f)
        index = 1
        for place in places:
            if index < starting_index:
                index += 1
                continue
            if index > max_index:
                break
            latitude = place["latitude"]
            longitude = place["longitude"]
            state = place["state"]
            prompt = f"What state does the place with latitude {latitude} and longitude {longitude} located in? Please give me your final answer in a new line at the end of your response."
            print(f"Generating ({index}/{len(places)}) places QA...")
            generate("hallucination-consistency/places", index, prompt, tokenizer, model, enable_thinking, do_sample, temperature)
            print(f"({index}/{len(places)}) places QA Done")
            index += 1

    ### historical figures
    with open("../../data/hallucination-consistency/historical_figures.json", "r") as f:
        figures = json.load(f)
        index = 1
        for figure in figures:
            if index < starting_index:
                index += 1
                continue
            if index > max_index:
                break
            description = figure["description"]
            birth_year = figure["birth_year"]
            death_year = figure["death_year"]
            prompt = f"What is the name of the historical figure who was born in {birth_year}, died in {death_year}, and is known for '{description}'? Please give me your final answer in a new line at the end of your response."
            print(f"Generating ({index}/{len(figures)}) historical_figures QA...")
            generate("hallucination-consistency/historical_figures", index, prompt, tokenizer, model, enable_thinking, do_sample, temperature)
            print(f"({index}/{len(figures)}) historical_figures QA Done")
            index += 1

    ### art
    with open("../../data/hallucination-consistency/art.json", "r") as f:
        art = json.load(f)
        index = 1
        for item in art:
            if index < starting_index:
                index += 1
                continue
            if index > max_index:
                break
            entity_type = item["entity_type"]
            release_date = item["release_date"]
            creator = item["creator"]
            prompt = f"What is the name of the famous {entity_type} that was released in {release_date} and created by {creator}? Please give me your final answer in a new line at the end of your response."
            print(f"Generating ({index}/{len(art)}) art QA...")
            generate("hallucination-consistency/art", index, prompt, tokenizer, model, enable_thinking, do_sample, temperature)
            print(f"({index}/{len(art)}) art QA Done")
            index += 1

# [x] 32B True 1~3
# [ ] 32B True 4~33
# [ ] 32B True 34~66
# [ ] 32B True 67~100