import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import pandas as pd
from basic_transformer import TOKENIZER_ITA_MODEL, TOKENIZER_NAP_MODEL
from basic_transformer import DATA_DIR


class TextDataset(Dataset):
    def __init__(self,
                 src_texts,
                 tgt_texts,
                 src_tokenizer,
                 tgt_tokenizer,
                 block_size):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer: spm.SentencePieceProcessor = src_tokenizer
        self.tgt_tokenizer: spm.SentencePieceProcessor = tgt_tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, index):
        src_text = self.src_texts[index]
        tgt_text = self.tgt_texts[index]
        input_ids = self.src_tokenizer.encode_as_ids(src_text)
        output_ids = [1] + self.tgt_tokenizer.encode_as_ids(tgt_text) + [2]

        # Truncate or pad the sequence
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        else:
            input_ids = input_ids + [0] * (self.block_size - len(input_ids))

        if len(output_ids) > self.block_size:
            output_ids = output_ids[:self.block_size]
        else:
            output_ids = output_ids + [0] * (self.block_size - len(output_ids))

        # Create target sequence for autoregressive prediction
        input_ids = torch.tensor(input_ids)
        target_ids = torch.tensor(output_ids)

        return input_ids, target_ids

# Load SentencePiece model


df = pd.read_csv(DATA_DIR / "dataset.csv")
total_size = df["italiano"].shape[0]
training_size = int(0.8 * total_size)
print(f"Total Size: {total_size}\nTraining Size: {training_size}")
italian_sencentes = df["italiano"].tolist()[:training_size]
neapolitan_sencentes = df["napoletano"].tolist()[:training_size]

sp_it = spm.SentencePieceProcessor(model_file=TOKENIZER_ITA_MODEL.as_posix())
sp_np = spm.SentencePieceProcessor(model_file=TOKENIZER_NAP_MODEL.as_posix())

# Initialize the dataset and DataLoader
dataset = TextDataset(italian_sencentes,
                      neapolitan_sencentes,
                      sp_it,
                      sp_np,
                      block_size=100)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
