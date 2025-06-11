import sys
import os
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


def generate(directory, file_index, prompt, client, model_name, thinking_budget, temperature, seed):
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

	# save the output to a file
	os.makedirs(f"./logs/{model_name}/thinking_{thinking_budget}/{directory}", exist_ok=True)
	with open(f"./logs/{model_name}/thinking_{thinking_budget}/{directory}/qa_{file_index}.txt", "w") as f:
		# write to the file
		f.write("**************************************** Inputs ****************************************\n")
		f.write("model: " + model_name + "\n")
		f.write("thinking_budget: " + str(thinking_budget) + "\n")
		f.write("temperature: " + str(temperature) + "\n")
		f.write("seed: " + str(seed) + "\n")
		f.write("prompt:\n" + prompt + "\n")
		f.write("*********************************** End of Inputs **************************************\n\n\n\n\n")

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
	temperature = 0.0
	seed = 10

	direct_prompt = " Show me your step-by-step reasoning process and the final answer (Yes or No)."
	prompts = [
        "Is a blue whale heavier than an elephant?",
        "Is Mount Everest taller than K2?",
        "Did Albert Einstein die later than Isaac Newton?",
        "Does the United States have a larger population than Canada?",
        "Is Russia larger in area than Australia?",
        "Is the Nile River longer than the Yangtze River?",
        "Is a cheetah faster than a lion?",
        "Is the Sahara Desert hotter than the Gobi Desert?",
        "Is Antarctica colder than the Arctic?",
        "Is gold more expensive than silver?",
        "Is the Pacific Ocean larger than the Atlantic Ocean?",
        "Did the Roman Empire exist before the British Empire?",
        "Is the Great Wall of China longer than the Berlin Wall?",
        "Does a blue whale have a larger heart than an elephant?",
        "Is the speed of light faster than the speed of sound?",
        "Is the boiling point of water higher than the freezing point of water?",
        "Does the Earth have more moons than Mars?",
        "Is the distance from Earth to the Sun greater than the distance from Earth to the Moon?",
        "Was the first moon landing after the first airplane flight?",
        "Is the atomic number of oxygen higher than that of carbon?",
        "Is Japan located east of China?",
        "Is Brazil located south of Canada?",
        "Is the pH of lemon juice lower than that of water?",
        "Does a day on Venus last longer than a day on Earth?",
        "Is the acceleration due to gravity on the Moon less than that on Earth?",
        "Was the discovery of penicillin before the discovery of DNA structure?",
        "Is the melting point of ice lower than the boiling point of water?",
        "Is a peregrine falcon faster than an eagle?",
        "Is the wavelength of red light longer than that of blue light?",
        "Is the average human lifespan longer than that of a dog?",
    ]

	index = 1
	for prompt in prompts:
		print(f"Generating ({index}/{len(prompts)}) QA...")
		generate("group_2-1", index, prompt + direct_prompt, client, model_name, thinking_budget, temperature, seed)
		print(f"({index}/{len(prompts)}) QA Done")
		index += 1

	prompts = [
        "Is an elephant heavier than a blue whale?",
        "Is K2 taller than Mount Everest?",
        "Did Isaac Newton die later than Albert Einstein?",
        "Does Canada have a larger population than the United States?",
        "Is Australia larger in area than Russia?",
        "Is the Yangtze River longer than the Nile River?",
        "Is a lion faster than a cheetah?",
        "Is the Gobi Desert hotter than the Sahara Desert?",
        "Is the Arctic colder than Antarctica?",
        "Is silver more expensive than gold?",
        "Is the Atlantic Ocean larger than the Pacific Ocean?",
        "Did the British Empire exist before the Roman Empire?",
        "Is the Berlin Wall longer than the Great Wall of China?",
        "Does an elephant have a larger heart than a blue whale?",
        "Is the speed of sound faster than the speed of light?",
        "Is the freezing point of water higher than the boiling point of water?",
        "Does Mars have more moons than the Earth?",
        "Is the distance from Earth to the Moon greater than the distance from Earth to the Sun?",
        "Was the first airplane flight after the first moon landing?",
        "Is the atomic number of carbon higher than that of oxygen?",
        "Is China located east of Japan?",
        "Is Canada located south of Brazil?",
        "Is the pH of water lower than that of lemon juice?",
        "Does a day on Earth last longer than a day on Venus?",
        "Is the acceleration due to gravity on Earth less than that on the Moon?",
        "Was the discovery of DNA structure before the discovery of penicillin?",
        "Is the boiling point of water lower than the melting point of ice?",
        "Is an eagle faster than a peregrine falcon?",
        "Is the wavelength of blue light longer than that of red light?",
        "Is the average lifespan of a dog longer than that of a human?",
    ]

	index = 1
	for prompt in prompts:
		print(f"Generating ({index}/{len(prompts)}) QA...")
		generate("group_2-2", index, prompt + direct_prompt, client, model_name, thinking_budget, temperature, seed)
		print(f"({index}/{len(prompts)}) QA Done")
		index += 1
