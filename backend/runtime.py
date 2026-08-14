import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
from generator import CustomGenerator

resource_id = os.path.join(os.path.expanduser("~/models"), "gpt2")

model = AutoModelForCausalLM.from_pretrained(resource_id)
tokenizer = AutoTokenizer.from_pretrained(resource_id)

generator = CustomGenerator(
    model = model,
    tokenizer = tokenizer
)

