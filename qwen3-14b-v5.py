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
    os.makedirs(f"./logs/{directory}", exist_ok=True)
    with open(f"./logs/{directory}/qa_{file_index}.txt", "w") as f:
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

    prompts = [
        "Is Peter Hitchens’s The Abolition of Britain shorter than Mo Hayder’s Birdman?",
        "Was J. M. Coetzee’s Summertime released later than Suzanne Collins’s Catching Fire?",
        "Was Elliot Silverstein’s Nightmare Honeymoon released later than Peter R. Hunt’s Gold?",
        "Is WILLIAMSBURG, Manhattan located south of HILLMAN HOUSES, Manhattan?",
        "Is RED BALLOON LEARNING CENTER, Manhattan located east of HUNTERS POINT S PARK PLAYGROUND, Queens?",
        "Was 'Clock Is Ticking for Recess, and for a Deficit Deal.' published earlier than 'Activist Running for Mayor of Moscow Is Briefly Detained.'?",
        "Did Anastasia of Sirmium live longer than William II, Count of Chalon?",
        "Was Yao Shu born earlier than Alfonso of Molina?",
        "Did Agnes of Aquitaine, Countess of Savoy die later than William de Warenne, 1st Earl of Surrey?",
        "Was Beastie Boys’s Jimmy James released earlier than Dream Theater’s Pull Me Under?",
        "Is Clymer, PA more densely populated than Monroe Manor, NJ?",
        "Is Flagler Estates, FL located south of Palatka, FL?",
        "Is Garnavillo, IA located east of Newellton, LA?",
        "Is Arbuckle, CA less populous than Pickens, SC?",
        "Is The University of Tampa, FL located north of Schiller International University, FL?",
        "Is The Ort Institute, IL located west of University of Wisconsin–Sheboygan, WI?",
        "Is Yamhill County, OR located north of Clackamas County, OR?",
        "Is Cambria County, PA located west of Cattaraugus County, NY?",
        "Is Eau Claire County, WI more populous than Platte County, MO?",
        "Is Lake Verret, LA located south of Calcasieu Lake, LA?",
        "Is Sangre de Cristo Range, CO located east of Sangre de Cristo Mountains, CO?",
        "Is Lake Martin, AL located south of Dal-Tex Building, TX?",
        "Is Pine Flat Dam, CA located east of St. Mary’s in the Mountains Catholic Church, NV?",
        "Is 83703, ID less densely populated than 79104, TX?",
        "Is 32751, FL located north of 32796, FL?",
        "Is 98580, WA located west of 97233, OR?",
        "Is 76234, TX more populous than 35473, AL?",
        "Does Þórisvatn have smaller area than Lake Tohopekaliga?",
        "Is Medina Lake located north of Rara Lake?",
        "Is Pragser Wildsee located west of Lake Vico?",
    ]

    index = 1
    for prompt in prompts:
        print(f"Generating ({index}/{len(prompts)}) QA...")
        generate("d_1", index, prompt, tokenizer, model, enable_thinking, do_sample, temperature)
        print(f"({index}/{len(prompts)}) QA Done")
        index += 1

    prompts = [
        "Is Mo Hayder’s Birdman shorter than Peter Hitchens’s The Abolition of Britain?",
        "Was Suzanne Collins’s Catching Fire released later than J. M. Coetzee’s Summertime?",
        "Was Peter R. Hunt’s Gold released later than Elliot Silverstein’s Nightmare Honeymoon?",
        "Is HILLMAN HOUSES, Manhattan located south of WILLIAMSBURG, Manhattan?",
        "Is HUNTERS POINT S PARK PLAYGROUND, Queens located east of RED BALLOON LEARNING CENTER, Manhattan?",
        "Was 'Activist Running for Mayor of Moscow Is Briefly Detained.' published earlier than 'Clock Is Ticking for Recess, and for a Deficit Deal.'?",
        "Did William II, Count of Chalon live longer than Anastasia of Sirmium?",
        "Was Alfonso of Molina born earlier than Yao Shu?",
        "Did William de Warenne, 1st Earl of Surrey die later than Agnes of Aquitaine, Countess of Savoy?",
        "Was Dream Theater’s Pull Me Under released earlier than Beastie Boys’s Jimmy James?",
        "Is Monroe Manor, NJ more densely populated than Clymer, PA?",
        "Is Palatka, FL located south of Flagler Estates, FL?",
        "Is Newellton, LA located east of Garnavillo, IA?",
        "Is Pickens, SC less populous than Arbuckle, CA?",
        "Is Schiller International University, FL located north of The University of Tampa, FL?",
        "Is University of Wisconsin–Sheboygan, WI located west of The Ort Institute, IL?",
        "Is Clackamas County, OR located north of Yamhill County, OR?",
        "Is Cattaraugus County, NY located west of Cambria County, PA?",
        "Is Platte County, MO more populous than Eau Claire County, WI?",
        "Is Calcasieu Lake, LA located south of Lake Verret, LA?",
        "Is Sangre de Cristo Mountains, CO located east of Sangre de Cristo Range, CO?",
        "Is Dal-Tex Building, TX located south of Lake Martin, AL?",
        "Is St. Mary’s in the Mountains Catholic Church, NV located east of Pine Flat Dam, CA?",
        "Is 79104, TX less densely populated than 83703, ID?",
        "Is 32796, FL located north of 32751, FL?",
        "Is 97233, OR located west of 98580, WA?",
        "Is 35473, AL less populous than 76234, TX?",
        "Does Lake Tohopekaliga have smaller area than Þórisvatn?",
        "Is Rara Lake located south of Medina Lake?",
        "Is Lake Vico located west of Pragser Wildsee?",
    ]

    index = 1
    for prompt in prompts:
        print(f"Generating ({index}/{len(prompts)}) QA...")
        generate("d_2", index, prompt, tokenizer, model, enable_thinking, do_sample, temperature)
        print(f"({index}/{len(prompts)}) QA Done")
        index += 1


# 30 Yes/No question pairs
# 1: the model gives Yes/Yes or No/No to the question pairs; 0: otherwise
# [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]