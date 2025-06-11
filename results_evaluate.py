import os
import sys
import re

log_dir = sys.argv[1]

result = []

# read the last 10 lines of the log file
log_files = os.listdir("./logs/" + log_dir)
log_files.sort(key=lambda x: int(x.split('_')[1][:-4]))
for log_file in log_files:
    with open(os.path.join("./logs/" + log_dir, log_file), 'r') as f:
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

print(result)