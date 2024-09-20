import sentencepiece as spm
from basic_transformer import DATA_DIR, TOKENIZER_DIR

vocab_size = 1000
options = {
    "input": DATA_DIR / "ita.txt",
    "input_format": "text",
    "model_prefix": TOKENIZER_DIR / f"tok_ita_{vocab_size}",
    "model_type": "bpe",
    "vocab_size": vocab_size,
    "character_coverage": 0.9995,
    "byte_fallback": True,
    "split_digits": True,
    "split_by_whitespace": True,
    "split_by_number": True,
    "split_by_unicode_script": True,
    "normalization_rule_name": "identity",
    "unk_id": 3,
    "eos_id": 2,
    "bos_id": 1,
    "pad_id": 0
}

if __name__ == "__main__":
    TRAIN = True
    phrase = "Ciao, come stai??"

    if TRAIN:
        spm.SentencePieceTrainer.train(**options)

    tokenizer = spm.SentencePieceProcessor(
        model_file=(DATA_DIR / "tokenizer" / f"tok_{vocab_size}.model").as_posix()
    )
    encoding = tokenizer.encode(phrase)
    decoding = tokenizer.decode(encoding)
    print(f"original: {phrase}\ndecoded: {decoding}")
