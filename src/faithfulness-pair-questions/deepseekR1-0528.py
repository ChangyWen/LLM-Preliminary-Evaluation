import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.environ['DEEPSEEK_API_KEY'],
  base_url="https://api.deepseek.com",
)
model_name = "deepseek-reasoner"

# load from system arguments
print("**************************************** Inputs ****************************************")
print("model:", model_name)

max_tokens = int(sys.argv[1])
print("max_tokens:", max_tokens)

prompt = sys.argv[2]
print("prompt:\n", prompt)

print("*********************************** End of Inputs **************************************\n\n\n\n\n")


# prepare the model input
messages = [
    {"role": "user", "content": prompt}
]
completion = client.chat.completions.create(
    model=model_name,
    max_tokens=max_tokens, # The maximum output length (including the COT part)
    messages=messages,
    seed=10,
)


print("*************************************** Thinking ***************************************")
print(completion.choices[0].message.reasoning_content)
print("*********************************** End of Thinking ************************************\n\n\n\n\n")

print("**************************************** Content ***************************************")
print(completion.choices[0].message.content)
print("************************************ End of Content ************************************")