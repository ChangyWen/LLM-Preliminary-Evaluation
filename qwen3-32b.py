import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

model_name = "Qwen/Qwen3-32B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


# load from system arguments
print("**************************************** Inputs ****************************************")
print("model:", model_name)

enable_thinking = sys.argv[1] == "True"
print("enable_thinking:", enable_thinking)

do_sample = False
temperature = 0.01
if len(sys.argv) > 3:
    do_sample = True
    temperature = float(sys.argv[3])

print("do_sample:", do_sample)
if do_sample:
    print("temperature:", temperature)

prompt = sys.argv[2]
print("prompt:\n", prompt)

print("*********************************** End of Inputs **************************************\n\n\n\n\n")


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

if enable_thinking:
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    print("*************************************** Thinking ***************************************")
    print(thinking_content)
    print("*********************************** End of Thinking ************************************\n\n\n\n\n")

content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print("**************************************** Content ***************************************")
print(content)
print("************************************ End of Content ************************************")



