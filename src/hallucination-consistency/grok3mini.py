import os
import sys
from openai import OpenAI
import json
from dotenv import load_dotenv
load_dotenv()


def generate(directory, file_index, prompt, client, model_name, thinking_level, temperature, seed):
    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model_name,
        reasoning_effort=thinking_level,
        messages=messages,
        temperature=temperature,
        seed=seed,
    )

    # save the output to a file
    os.makedirs(f"../../logs/{directory}/{model_name}/thinking_{thinking_level}", exist_ok=True)
    with open(f"../../logs/{directory}/{model_name}/thinking_{thinking_level}/qa_{file_index}.txt", "w") as f:
        # write to the file
        f.write("**************************************** Inputs ****************************************\n")
        f.write("model: " + model_name + "\n")
        f.write("thinking_level: " + str(thinking_level) + "\n")
        f.write("temperature: " + str(temperature) + "\n")
        f.write("seed: " + str(seed) + "\n")
        f.write("prompt:\n" + prompt + "\n")
        f.write("*********************************** End of Inputs **************************************\n\n\n\n\n")

        f.write("*************************************** Thinking ***************************************\n")
        f.write(completion.choices[0].message.reasoning_content + "\n")
        f.write("*********************************** End of Thinking ************************************\n\n\n\n\n")

        f.write("**************************************** Content ***************************************\n")
        f.write(completion.choices[0].message.content + "\n")
        f.write("************************************ End of Content ************************************\n\n\n\n\n")


if __name__ == "__main__":
    client = OpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url="https://api.x.ai/v1",
    )
    model_name = "grok-3-mini"
    thinking_level = sys.argv[1]
    starting_index = int(sys.argv[2])
    max_index = int(sys.argv[3])
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
            generate("hallucination-consistency/places", index, prompt, client, model_name,thinking_level, temperature, seed)
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
            generate("hallucination-consistency/historical_figures", index, prompt, client, model_name,thinking_level, temperature, seed)
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
            generate("hallucination-consistency/art", index, prompt, client, model_name,thinking_level, temperature, seed)
            print(f"({index}/{len(art)}) art QA Done")
            index += 1
