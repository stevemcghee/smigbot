# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="google/gemma-2-9b", filename="config.json") # not needed?

# mps_device = torch.device("mps")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("mps")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
 