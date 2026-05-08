import torch

try:
    from embeddings.model_loader import tokenizer, model, device
except ImportError:
    from model_loader import tokenizer, model, device

def encode(instructions: list[str]):
    func_dict = {str(i): instr for i, instr in enumerate(instructions)}
    inputs = tokenizer([func_dict], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model(**inputs)  # shape: (1, hidden_dim), already normalized

    return embedding.squeeze(0).cpu().numpy()