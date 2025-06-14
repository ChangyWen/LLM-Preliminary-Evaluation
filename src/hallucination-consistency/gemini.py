import sys
import os
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
import json
from dotenv import load_dotenv

load_dotenv()


def generate(directory, file_index, prompt, client, model_name, thinking_budget, temperature, seed):
    # save the output to a file
    os.makedirs(f"../../logs/{directory}/{model_name}/thinking_{thinking_budget}", exist_ok=True)
    # check if the file exists
    if os.path.exists(f"../../logs/{directory}/{model_name}/thinking_{thinking_budget}/qa_{file_index}.txt"):
        print(f"Error: file already exists; file_index: {file_index}")
        return

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True if thinking_budget > 0 else False,
            ),
            temperature=temperature,
            seed=seed
        )
    )

    with open(f"../../logs/{directory}/{model_name}/thinking_{thinking_budget}/qa_{file_index}.txt", "w") as f:
        # write to the file
        f.write("**************************************** Inputs ****************************************\n")
        f.write("model: " + model_name + "\n")
        f.write("thinking_budget: " + str(thinking_budget) + "\n")
        f.write("temperature: " + str(temperature) + "\n")
        f.write("seed: " + str(seed) + "\n")
        f.write("prompt:\n" + prompt + "\n")
        f.write("*********************************** End of Inputs **************************************\n\n\n\n\n")

        # check if response.candidates is None
        if response.candidates is None:
            print(f"Error: response.candidates is None; file_index: {file_index}")
            return
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                f.write("*************************************** Thinking ***************************************\n")
                f.write(part.text + "\n")
                f.write("*********************************** End of Thinking ************************************\n\n\n\n\n")
            else:
                f.write("**************************************** Content ***************************************\n")
                f.write(part.text + "\n")
                f.write("************************************ End of Content ************************************\n\n\n\n\n")
                break


if __name__ == "__main__":
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    model_type = sys.argv[1]
    model_name = "gemini-2.5-"
    if model_type == "flash":
        model_name += "flash-preview-05-20"
    else:
        model_name += "pro-preview-06-05"

    thinking_budget = int(sys.argv[2])
    if model_type == "flash":
        assert thinking_budget <= 24576 and thinking_budget >= 0
    else:
        assert thinking_budget <= 32768 and thinking_budget >= 128

    starting_index = int(sys.argv[3])
    max_index = int(sys.argv[4])
    temperature = 0.0
    seed = 10

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
            generate("hallucination-consistency/places", index, prompt, client, model_name, thinking_budget, temperature, seed)
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
            generate("hallucination-consistency/historical_figures", index, prompt, client, model_name, thinking_budget, temperature, seed)
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
            generate("hallucination-consistency/art", index, prompt, client, model_name, thinking_budget, temperature, seed)
            print(f"({index}/{len(art)}) art QA Done")
            index += 1


