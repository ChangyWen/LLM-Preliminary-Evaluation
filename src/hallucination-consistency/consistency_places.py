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


with open("../../logs/hallucination-consistency/art/compared_results.json", "r") as f:
    compared_results = json.load(f)
    index = 1
    hallucinated_item_count = 0
    multi_hallucinated_item_count = 0
    multi_hallucinated_items = []
    while index <= 100:
        if index == 33 or index == 75 or index == 76 or index == 78 or index == 79:
            index += 1
            continue
        item = compared_results[f"{index}"]
        index += 1

        answers = item["answers"]
        labels = item["labels"]
        if "0" not in labels: continue
        print(index - 1)
        # print(answers)
        # print(labels)
        print(f"**** {item['ground_truth']} ****")
        hallucinated_item_count += 1
        indices_of_hallucinated_item = [i for i, x in enumerate(labels) if x == "0"]
        if len(indices_of_hallucinated_item) > 1:
            multi_hallucinated_item_count += 1
            multi_hallucinated_items.append(len(indices_of_hallucinated_item))
        for idx in indices_of_hallucinated_item:
            print(f"[{answers[idx]}]")
        print("\n\n")
    print(f"hallucinated item count: {hallucinated_item_count}")
    print(f"multi-hallucinated item count: {multi_hallucinated_item_count}")
    print(f"multi-hallucinated items: {multi_hallucinated_items}")

# art 1
# 89 (2)

# historical figures 15
# 17 (3), 27 (2), 30 (4), 63 (2)

# art 11
# 2 (2), 15 (3)



# ***** Preliminary evaluation settings *****

# 6个模型：DeepSeek-R1-0528,   Gemini-2.5-Flash-Preview-05-20,   Gemini-2.5-Pro-Preview-06-05,   Grok-3-Mini,   Qwen3-14B,   Qwen3-32B。

# 三个领域的知识：地理、历史人物（wiki有记录的）、文艺作品（包括书、电影、音乐），每个领域100个问题。

# 地理领域的问题：给出一个经纬度，问该经纬度的地点位于哪个US state。
# 历史人物领域的问题：给出某历史人物的出生、离世年份 & 关于该历史人物的一句话描述，问该历史人物的名字。
# 文艺作品领域的问题：给出发布日期、作者、以及作品类别（书、电影、or 音乐），问该文艺作品的名字。



# ***** Preliminary results *****

# - 地理领域：
# 6 (out of 100) 个问题的回答里出现hallucinations，
# 其中 5 (out of 6) 个问题只有单个模型出现hallucinations，
# 其他 1 (out of 6) 个问题有 2 个模型出现hallucinations，
# 其中 1 (out of 1) 个问题的回答里有 2 个模型给出consistent/same hallucinations;

# - 历史人物领域：
# 21 (out of 100) 个问题的回答里出现hallucinations，
# 其中 6 (out of 21) 个问题只有单个模型出现hallucinations，
# 其他 15 (out of 21) 个问题有 3.2 (in average) 个模型出现hallucinations，
# 其中 4 (out of 15) 个问题的回答里有 2.75 (in average) 个模型给出consistent/same hallucinations;

# - 文艺作品领域：
# 34 (out of 100) 个问题的回答里出现hallucinations，
# 其中 21 (out of 34) 个问题只有单个模型出现hallucinations，
# 其他 13 (out of 34) 个问题有 3.46 (in averge) 个模型出现hallucinations，
# 其中 4 (out of 13) 个问题的回答里有 3 (in average) 个模型给出consistent/same hallucinations;