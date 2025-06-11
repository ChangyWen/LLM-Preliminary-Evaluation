import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.environ['XAI_API_KEY'],
  base_url="https://api.x.ai/v1",
)
model_name = "grok-3-mini"

# load from system arguments
print("**************************************** Inputs ****************************************")
print("model:", model_name)

thinking_level = sys.argv[1]
print("thinking_level:", thinking_level)

prompt = sys.argv[2]
print("prompt:\n", prompt)

print("*********************************** End of Inputs **************************************\n\n\n\n\n")


# prepare the model input
messages = [
    {"role": "user", "content": prompt}
]
completion = client.chat.completions.create(
    model=model_name,
    reasoning_effort=thinking_level,
    messages=messages,
    temperature=0.0,
    seed=10,
)


print("*************************************** Thinking ***************************************")
print(completion.choices[0].message.reasoning_content)
print("*********************************** End of Thinking ************************************\n\n\n\n\n")

print("**************************************** Content ***************************************")
print(completion.choices[0].message.content)
print("************************************ End of Content ************************************")