import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

deepseek_dir = "DeepSeek-R1-0528/max_tokens_12000"
gemini_flash_dir = "gemini-2.5-flash-preview-05-20/thinking_24576"
gemini_pro_dir = "gemini-2.5-pro-preview-06-05/thinking_32768"
grok3mini_dir = "grok-3-mini/thinking_high"
qwen3_14b_dir = "Qwen3-14B/thinking_True"
qwen3_32B_dir = "Qwen3-32B/thinking_True"
all_dirs = [deepseek_dir, gemini_flash_dir, gemini_pro_dir, grok3mini_dir, qwen3_14b_dir, qwen3_32B_dir]

client = OpenAI(
  api_key=os.environ['XAI_API_KEY'],
  base_url="https://api.x.ai/v1",
)

def compare(entity_type, ground_truth, answer):
    messages = [
        {"role": "user", "content": f"Please compare the following answer and the ground truth regarding the {entity_type} title (do not need to be exact match). The answer is '{answer}'. The ground truth is '{ground_truth}'. Response only '1' if the answer mention the ground-truth {entity_type} title, otherwise response only '0' (no explanation needed)."}
    ]
    completion = client.chat.completions.create(
        model="grok-3-latest",
        messages=messages,
        temperature=0.0,
        seed=10,
    )
    if len(completion.choices) == 0:
        return "N/A"
    else:
        return completion.choices[0].message.content.strip()


# places
with open("../../data/hallucination-consistency/art.json", "r") as f:
    compared_results = {}
    arts = json.load(f)
    index = 1
    for item in arts:
        if index > 100: break
        compared_results[index] = { "answers": [], "labels": [] }
        print(f"Processing place {index}...")
        name = item["title"]
        for dir in all_dirs:
            with open(f"../../logs/hallucination-consistency/art/{dir}/qa_{index}.txt", "r") as f:
                # remove all the empty lines at the end of the file
                lines = f.readlines()
                while lines[-1] == "\n":
                    lines.pop()
                lines.pop()
                tmp1 = lines.pop()
                tmp2 = lines.pop()
                answer = ""
                if tmp2 == "**************************************** Content ***************************************\n":
                    answer = tmp1
                else:
                    answer = tmp2 + tmp1
                # change all new lines to spaces
                answer = answer.replace("\n", " ")
                result = compare("art work", name, answer)
                print(f"ground-truth: {name}, answer: {answer}, result: {result}")
                compared_results[index]["answers"].append(answer)
                compared_results[index]["labels"].append(result)
        index += 1
    with open("../../logs/hallucination-consistency/art/compared_results.json", "w") as f:
        json.dump(compared_results, f, indent=4)


