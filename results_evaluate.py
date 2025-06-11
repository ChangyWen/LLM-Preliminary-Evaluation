import os
import sys
import re

log_dirs = [
    "grok-3-mini/thinking_high/",
    "grok-3-mini/thinking_low/",
    "Qwen3-14B/thinking_False/",
    "Qwen3-14B/thinking_True/",
    "DeepSeek-R1-0528/max_tokens_32000/",
]

for log_dir in log_dirs:
    for group in ["group_1-1", "group_1-2", "group_2-1", "group_2-2"]:
        log_dir_group = log_dir + group + "/"
        result = []

        # read the last 10 lines of the log file
        log_files = os.listdir("./logs/" + log_dir_group)
        log_files.sort(key=lambda x: int(x.split('_')[1][:-4]))
        for log_file in log_files:
            with open(os.path.join("./logs/" + log_dir_group, log_file), 'r') as f:
                log_lines = f.readlines()[-15:]
                all_lines_str = "".join(log_lines)
                isYes = bool(re.search(r'\bYES\b', all_lines_str.upper()))
                isNo = bool(re.search(r'\bNO\b', all_lines_str.upper()))
                if isYes and not isNo:
                    result.append(1)
                elif isNo and not isYes:
                    result.append(0)
                else:
                    result.append(-1)

        print(log_dir_group)
        print(result)
        print("\n\n")