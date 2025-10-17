
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_dataloader(tokenizer, batch_size=2, max_length=64):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids'])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)
