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

def compare(entity_type, answer1, answer2):
    messages = [
        {"role": "user", "content": f"Please compare the following two answers. The first answer is '{answer1}'. The second answer is '{answer2}'. Respond only '1' if the two answers point to the same {entity_type}, otherwise respond only '0' (no explanation needed)."}
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


with open("../../logs/hallucination-consistency/places/compared_results.json", "r") as f:
    compared_results = json.load(f)
    index = 1
    hallucinated_item_count = 0
    while index <= 100:
        item = compared_results[f"{index}"]
        index += 1

        answers = item["answers"]
        labels = item["labels"]
        if "0" not in labels: continue
        print(index - 1)
        print(answers)
        print(labels)
        print(f"**** {item['ground_truth']} ****")
        hallucinated_item_count += 1
        indices_of_hallucinated_item = [i for i, x in enumerate(labels) if x == "0"]
        for idx in indices_of_hallucinated_item:
            print(f"[{answers[idx]}]")
        print("\n\n")
    print(f"hallucinated item count: {hallucinated_item_count}")



