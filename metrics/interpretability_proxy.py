
import torch

def compute_attention_entropy(model, dataloader, device):
    model.eval()
    entropies = []
    with torch.no_grad():
        for batch in list(dataloader)[:5]:
            inputs = batch['input_ids'].to(device)
            attn = model(inputs, output_attentions=True).attentions
            layer_ent = [(-a * a.log()).sum(dim=-1).mean().item() for a in attn]
            entropies.append(sum(layer_ent) / len(layer_ent))
    model.train()
    return sum(entropies) / len(entropies)
