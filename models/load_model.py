
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name='distilgpt2', device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
