import torch

def get_text_transform(tokenizer, max_len=30):
    def transform(caption):
        tokens = tokenizer(caption)
        ids = tokens[:max_len]
        ids += [0] * (max_len - len(ids))  # padding
        return torch.tensor(ids, dtype=torch.long)
    return transform
