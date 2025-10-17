
import torch
import numpy as np

def evaluate_calibration(model, dataloader, tokenizer, device):
    model.eval()
    nlls, confs = [], []
    with torch.no_grad():
        for batch in list(dataloader)[:5]:
            inputs = batch['input_ids'].to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss.item()
            nlls.append(loss)
            probs = torch.softmax(outputs.logits, dim=-1)
            confs.append(probs.max().item())
    model.train()
    ece = np.abs(np.mean(confs) - (1 - np.mean(nlls) / 10))
    return ece, np.mean(nlls)
