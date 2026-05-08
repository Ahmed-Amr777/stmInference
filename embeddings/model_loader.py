from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch

BASE = "hustcw/clap-asm"
ADAPTER = "models/finetuned/lora_adapters"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

base_model = AutoModel.from_pretrained(BASE, trust_remote_code=True)

model = PeftModel.from_pretrained(base_model, ADAPTER)

model = model.merge_and_unload()

model.to(device)
model.eval()