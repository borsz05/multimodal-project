import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple
from transformers import BertTokenizerFast

import torch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def basic_tokenize(text: str) -> List[str]: #Splits text into lowercase word tokens
    return re.findall(r"\w+", text.lower())


def build_vocab( #Counts word frequencies and builds a vocab with PAD/UNK tokens
    captions: Iterable[str],
    min_freq: int = 2,
    max_size: int | None = None,
) -> Dict[str, int]:
    counter = Counter()
    for cap in captions:
        counter.update(basic_tokenize(cap))

    vocab_tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab_tokens.sort(key=lambda x: (-counter[x], x))
    if max_size is not None:
        vocab_tokens = vocab_tokens[: max_size - 2]  # reserve for PAD/UNK

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok in vocab_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


class VocabTokenizer: #Converts text tokens into vocab IDs
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.unk_id = vocab.get(UNK_TOKEN, 1)

    def __call__(self, text: str) -> List[int]:
        tokens = basic_tokenize(text)
        return [self.vocab.get(tok, self.unk_id) for tok in tokens]


def get_text_transform(tokenizer: VocabTokenizer, max_len: int = 30): #Turns caption into padded ID list + attention mask
    """
    Returns a function that converts caption -> (token_ids, attention_mask)
    - token_ids: LongTensor [L]
    - attention_mask: FloatTensor [L], 1 for real tokens, 0 for padding
    """

    def transform(caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = tokenizer(caption)
        ids = ids[:max_len]
        attn = [1.0] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [0] * pad_len
            attn += [0.0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.float32)

    return transform

def get_bert_text_transform(tokenizer: BertTokenizerFast, max_len: int = 32):
    """
    BERT tokenization:
    caption -> (input_ids, attention_mask)
    - input_ids: LongTensor [L]
    - attention_mask: FloatTensor [L]
    """
    def transform(caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attn_mask = encoded["attention_mask"].squeeze(0).float()
        return input_ids, attn_mask

    return transform