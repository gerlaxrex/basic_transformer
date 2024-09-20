import os
import pathlib as pl

__BASE_DIR = pl.Path(os.path.dirname(__file__))
DATA_DIR = __BASE_DIR / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
TOKENIZER_ITA_MODEL = TOKENIZER_DIR / "tok_ita_10000.model"
TOKENIZER_NAP_MODEL = TOKENIZER_DIR / "tok_nap_10000.model"
