import os
import sys
import re
import json

configs = [
    ["grok-3-mini", "thinking_high"],
    ["grok-3-mini", "thinking_low"],
    ["gemini-2.5-flash-preview-05-20", "thinking_0"],
    ["gemini-2.5-flash-preview-05-20", "thinking_640"],
    ["gemini-2.5-flash-preview-05-20", "thinking_24576"],
    ["gemini-2.5-pro-preview-06-05", "thinking_640"],
    ["gemini-2.5-pro-preview-06-05", "thinking_32768"],
    ["Qwen3-14B", "thinking_False"],
    ["Qwen3-14B", "thinking_True"],
    ["DeepSeek-R1-0528", "max_tokens_32000"],
    ["DeepSeek-R1-0528", "max_tokens_4096"],
    ["Qwen3-32B", "thinking_False"],
    ["Qwen3-32B", "thinking_True"],
]
results = json.load(open("./logs/results.json", "r"))

for config in configs:
    model = config[0]
    thinking = config[1]
    group_1_1 = results[model][thinking]["group_1-1"]
    group_1_2 = results[model][thinking]["group_1-2"]
    yy_count = 0
    nn_count = 0
    for i in range(30):
        if i == 28:
            if not(group_1_1[i] == group_1_2[i]):
                yy_count += 1
        else:
            if group_1_1[i] == 1 and group_1_2[i] == 1:
                yy_count += 1
            elif group_1_1[i] == 0 and group_1_2[i] == 0:
                nn_count += 1
    print("long-tail", model, thinking, yy_count + nn_count, (yy_count + nn_count) / 30)


    group_2_1 = results[model][thinking]["group_2-1"]
    group_2_2 = results[model][thinking]["group_2-2"]
    yy_count = 0
    nn_count = 0
    for i in range(30):
        if group_2_1[i] == 1 and group_2_2[i] == 1:
            yy_count += 1
        elif group_2_1[i] == 0 and group_2_2[i] == 0:
            nn_count += 1
    print("non long-tail", model, thinking, yy_count + nn_count, (yy_count + nn_count) / 30)
    print("\n\n")


