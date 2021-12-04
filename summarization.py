import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from files import pickle_load

model_name = "IlyaGusev/rugpt3medium_sum_gazeta"

auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
model_casual_lm = AutoModelForCausalLM.from_pretrained(model_name)


# from text_embeddings import SlidingEmbedder


# https://huggingface.co/IlyaGusev/rugpt3medium_sum_gazeta
def do_summarize(article_text):
    '''
    Берем GPT от сбера, по скользящему окну строим саммари по предложениям.
    '''
    text_tokens = auto_tokenizer(
        article_text,
        max_length=600,
        add_special_tokens=False,
        padding=False,
        truncation=True
    )["input_ids"]
    input_ids = text_tokens + [auto_tokenizer.sep_token_id]
    input_ids = torch.LongTensor([input_ids])

    output_ids = model_casual_lm.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )

    summary = auto_tokenizer.decode(output_ids[0], skip_special_tokens=False)
    summary = summary.split(auto_tokenizer.sep_token)[1]
    summary = summary.split(auto_tokenizer.eos_token)[0]
    return summary

# def get_full_text(tokens_list):
#     return ' '.join(token['word'] for token in tokens_list)
#
#
# print(datetime.now())
# results = pickle_load('processed_result.gz.mdl')
# full_text = get_full_text(results['tokens'][0])
# print(full_text)
# tokens_df = pd.DataFrame(results['tokens'][0])

# embedder_model = SlidingEmbedder(embedder=do_summarize)
# sentiments = embedder_model.make_embeddings(tokens_df.start, tokens_df.word)
# print(sentiments)
# # print(full_text)
# # print(summarize(full_text))
# print(datetime.now())
