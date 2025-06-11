import sys
import os
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

model_type = sys.argv[1]
model_name = "gemini-2.5-"
if model_type == "flash":
    model_name += "flash-preview-05-20"
else:
    model_name += "pro-preview-06-05"

thinking_budget = int(sys.argv[2])
if model_type == "flash":
    assert thinking_budget <= 24576 and thinking_budget >= 0
else:
    assert thinking_budget <= 32768 and thinking_budget >= 128

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model=model_name,
    contents="How does AI work?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
           thinking_budget=thinking_budget,
           include_thoughts=True if thinking_budget > 0 else False,
        ),
        temperature=0.0,
        seed=10
    )
)


for part in response.candidates[0].content.parts:
  if not part.text:
    continue
  if part.thought:
    print("*************************************** Thinking ***************************************")
    print(part.text)
    print("*********************************** End of Thinking ************************************\n\n\n\n\n")
  else:
    print("**************************************** Content ***************************************")
    print(part.text)
    print("************************************ End of Content ************************************\n\n\n\n\n")
    break