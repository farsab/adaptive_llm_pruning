
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.wikitext_loader import get_dataloader
from pruning.adaptive_pruner import AdaptivePruner
from pruning.reactivation_burst import ReactivationBurst
from metrics.calibration import evaluate_calibration
from metrics.interpretability_proxy import compute_attention_entropy
from metrics.logger import MetricsLogger
from tqdm import tqdm

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--target_sparsity', type=float, default=0.4)
    parser.add_argument('--reactivate_every', type=int, default=200)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    dataloader = get_dataloader(tokenizer)

    pruner = AdaptivePruner(model, target_sparsity=args.target_sparsity)
    reactivator = ReactivationBurst(model)
    logger = MetricsLogger()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    step = 0

    model.train()
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs = batch['input_ids'].to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()

            pruner.update_masks(model)  # adaptive pruning
            optimizer.step()
            optimizer.zero_grad()

            if step % args.reactivate_every == 0 and step > 0:
                reactivator.reactivate(model)

            if step % 100 == 0:
                ece, nll = evaluate_calibration(model, dataloader, tokenizer, device)
                entropy = compute_attention_entropy(model, dataloader, device)
                logger.log(step, ece, nll, entropy)

            step += 1

    logger.save_plots()
    print("Training + adaptive pruning completed. Results saved in results/.")

if __name__ == "__main__":
    main()
