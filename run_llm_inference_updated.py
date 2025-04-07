import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# model
model_name = "google/gemma-2b-it"
token = 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, use_auth_token=token)
model.to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device

# read data
df = pd.read_csv("LLM_Stance.csv")

# prompt with few-shot examples
def make_prompt(post_text):
    return f"""You are a stance classification model. Classify the stance expressed in a Reddit post regarding the U.S. role in Ukraine.

Possible labels: Favor, Oppose, Neutral, Irrelevant

Examples:

Post: "Tens of thousands of anti-government protesters hold rally ahead of Romania's election rerun."
Label: Irrelevant

Post: "UK-France tensions over plan to seize $350bn Russia assets for US arms | Ukraine."
Label: Favor

Post: "A weekend of frantic talks - where does it leave Zelensky, Trump and Europe."
Label: Neutral

Post: "Allies Must Work With Zelenskiy After Spat With Trump, Canadas Joly Says."
Label: Oppose

Now classify the following post:
Post: "{post_text}"  
Label:"""

# label extractor
def extract_label(output_text):
    for label in ["Favor", "Oppose", "Neutral", "Irrelevant"]:
        if label.lower() in output_text.lower():
            return label
    return "Unknown"

# run inference
def get_prediction(text):
    prompt = make_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_label(result)

# apply to combined column
tqdm.pandas(desc="Predicting with Gemma")
df["local_llm_pred"] = df["combined"].progress_apply(get_prediction)

# save output
df.to_csv("LLM_Predictions_Output.csv", index=False)
