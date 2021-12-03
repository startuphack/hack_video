from typing import List

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
# python -m dostoevsky download fasttext-social-network-model

tokenizer = RegexTokenizer()

model = FastTextSocialNetworkModel(tokenizer=tokenizer)


def get_sentinent(text: str):
    return model.predict([text], k=2)[0]
