import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

def generate(directory, file_index, prompt, tokenizer, model, enable_thinking, do_sample, temperature):
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

    # save the output to a file
    os.makedirs(f"./logs/thinking_{enable_thinking}/{directory}", exist_ok=True)
    with open(f"./logs/thinking_{enable_thinking}/{directory}/qa_{file_index}.txt", "w") as f:
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
        f.write("************************************ End of Content ************************************\n\n\n\n\n")


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-14B"
    enable_thinking = sys.argv[1] == "True"
    do_sample = False
    temperature = 0.01
    if len(sys.argv) > 2:
        do_sample = True
        temperature = float(sys.argv[2])

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

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
        generate("group_1-1", index, prompt + direct_prompt, tokenizer, model, enable_thinking, do_sample, temperature)
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
        generate("group_1-2", index, prompt + direct_prompt, tokenizer, model, enable_thinking, do_sample, temperature)
        print(f"({index}/{len(prompts)}) QA Done")
        index += 1


# 30 Yes/No question pairs (non-thinking, autodl)
# 1: the model gives Yes/Yes or No/No to the question pairs; 0: otherwise
# [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]