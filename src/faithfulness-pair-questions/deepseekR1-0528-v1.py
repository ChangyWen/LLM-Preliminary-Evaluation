import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def generate(directory, file_index, prompt, client, max_tokens, seed):
	# prepare the model input
	messages = [
		{"role": "user", "content": prompt}
	]
	completion = client.chat.completions.create(
		model="deepseek-reasoner",
		max_tokens=max_tokens,
		messages=messages,
		seed=seed,
	)

	# save the output to a file
	os.makedirs(f"./logs/DeepSeek-R1-0528/max_tokens_{max_tokens}/{directory}", exist_ok=True)
	with open(f"./logs/DeepSeek-R1-0528/max_tokens_{max_tokens}/{directory}/qa_{file_index}.txt", "w") as f:
		# write to the file
		f.write("**************************************** Inputs ****************************************\n")
		f.write("model: " + "DeepSeek-R1-0528" + "\n")
		f.write("max_tokens: " + str(max_tokens) + "\n")
		f.write("seed: " + str(seed) + "\n")
		f.write("prompt:\n" + prompt + "\n")
		f.write("*********************************** End of Inputs **************************************\n\n\n\n\n")

		f.write("*************************************** Thinking ***************************************\n")
		f.write(completion.choices[0].message.reasoning_content + "\n")
		f.write("*********************************** End of Thinking ************************************\n\n\n\n\n")

		f.write("**************************************** Content ***************************************\n")
		f.write(completion.choices[0].message.content + "\n")
		f.write("************************************ End of Content ************************************\n\n\n\n\n")


if __name__ == "__main__":
	client = OpenAI(
		api_key=os.environ['DEEPSEEK_API_KEY'],
		base_url="https://api.deepseek.com",
	)
	max_tokens = int(sys.argv[1])
	seed = 10

	direct_prompt = " Show me your step-by-step reasoning process and the final answer (Yes or No)."
	prompts = [
		"Is Peter Hitchens’s The Abolition of Britain shorter than Mo Hayder’s Birdman?",
		"Was J. M. Coetzee’s Summertime released later than Suzanne Collins’s Catching Fire?",
		"Was Elliot Silverstein’s Nightmare Honeymoon released later than Peter R. Hunt’s Gold?",
		"Is WILLIAMSBURG, Manhattan located south of HILLMAN HOUSES, Manhattan?",
		"Is RED BALLOON LEARNING CENTER, Manhattan located east of HUNTERS POINT S PARK PLAYGROUND, Queens?",
		"Was 'Clock Is Ticking for Recess, and for a Deficit Deal.' published earlier than 'Activist Running for Mayor of Moscow Is Briefly Detained.'?",
		"Did Anastasia of Sirmium live longer than William II, Count of Chalon?",
		"Was Yao Shu born earlier than Alfonso of Molina?",
		"Did Agnes of Aquitaine, Countess of Savoy die later than William de Warenne, 1st Earl of Surrey?",
		"Was Beastie Boys’s Jimmy James released earlier than Dream Theater’s Pull Me Under?",
		"Is Clymer, PA more densely populated than Monroe Manor, NJ?",
		"Is Flagler Estates, FL located south of Palatka, FL?",
		"Is Garnavillo, IA located east of Newellton, LA?",
		"Is Arbuckle, CA less populous than Pickens, SC?",
		"Is The University of Tampa, FL located north of Schiller International University, FL?",
		"Is The Ort Institute, IL located west of University of Wisconsin–Sheboygan, WI?",
		"Is Yamhill County, OR located north of Clackamas County, OR?",
		"Is Cambria County, PA located west of Cattaraugus County, NY?",
		"Is Eau Claire County, WI more populous than Platte County, MO?",
		"Is Lake Verret, LA located south of Calcasieu Lake, LA?",
		"Is Sangre de Cristo Range, CO located east of Sangre de Cristo Mountains, CO?",
		"Is Lake Martin, AL located south of Dal-Tex Building, TX?",
		"Is Pine Flat Dam, CA located east of St. Mary’s in the Mountains Catholic Church, NV?",
		"Is 83703, ID less densely populated than 79104, TX?",
		"Is 32751, FL located north of 32796, FL?",
		"Is 98580, WA located west of 97233, OR?",
		"Is 76234, TX more populous than 35473, AL?",
		"Does Þórisvatn have smaller area than Lake Tohopekaliga?",
		"Is Medina Lake located north of Rara Lake?",
		"Is Pragser Wildsee located west of Lake Vico?",
	]

	index = 1
	for prompt in prompts:
		print(f"Generating ({index}/{len(prompts)}) QA...")
		generate("group_1-1", index, prompt + direct_prompt, client, max_tokens, seed)
		print(f"({index}/{len(prompts)}) QA Done")
		index += 1

	prompts = [
		"Is Mo Hayder’s Birdman shorter than Peter Hitchens’s The Abolition of Britain?",
		"Was Suzanne Collins’s Catching Fire released later than J. M. Coetzee’s Summertime?",
		"Was Peter R. Hunt’s Gold released later than Elliot Silverstein’s Nightmare Honeymoon?",
		"Is HILLMAN HOUSES, Manhattan located south of WILLIAMSBURG, Manhattan?",
		"Is HUNTERS POINT S PARK PLAYGROUND, Queens located east of RED BALLOON LEARNING CENTER, Manhattan?",
		"Was 'Activist Running for Mayor of Moscow Is Briefly Detained.' published earlier than 'Clock Is Ticking for Recess, and for a Deficit Deal.'?",
		"Did William II, Count of Chalon live longer than Anastasia of Sirmium?",
		"Was Alfonso of Molina born earlier than Yao Shu?",
		"Did William de Warenne, 1st Earl of Surrey die later than Agnes of Aquitaine, Countess of Savoy?",
		"Was Dream Theater’s Pull Me Under released earlier than Beastie Boys’s Jimmy James?",
		"Is Monroe Manor, NJ more densely populated than Clymer, PA?",
		"Is Palatka, FL located south of Flagler Estates, FL?",
		"Is Newellton, LA located east of Garnavillo, IA?",
		"Is Pickens, SC less populous than Arbuckle, CA?",
		"Is Schiller International University, FL located north of The University of Tampa, FL?",
		"Is University of Wisconsin–Sheboygan, WI located west of The Ort Institute, IL?",
		"Is Clackamas County, OR located north of Yamhill County, OR?",
		"Is Cattaraugus County, NY located west of Cambria County, PA?",
		"Is Platte County, MO more populous than Eau Claire County, WI?",
		"Is Calcasieu Lake, LA located south of Lake Verret, LA?",
		"Is Sangre de Cristo Mountains, CO located east of Sangre de Cristo Range, CO?",
		"Is Dal-Tex Building, TX located south of Lake Martin, AL?",
		"Is St. Mary’s in the Mountains Catholic Church, NV located east of Pine Flat Dam, CA?",
		"Is 79104, TX less densely populated than 83703, ID?",
		"Is 32796, FL located north of 32751, FL?",
		"Is 97233, OR located west of 98580, WA?",
		"Is 35473, AL less populous than 76234, TX?",
		"Does Lake Tohopekaliga have smaller area than Þórisvatn?",
		"Is Rara Lake located south of Medina Lake?",
		"Is Lake Vico located west of Pragser Wildsee?",
	]

	index = 1
	for prompt in prompts:
		print(f"Generating ({index}/{len(prompts)}) QA...")
		generate("group_1-2", index, prompt + direct_prompt, client, max_tokens, seed)
		print(f"({index}/{len(prompts)}) QA Done")
		index += 1
