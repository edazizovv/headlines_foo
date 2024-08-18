from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id)  # .to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# from datasets import load_dataset

# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# encodings = tokenizer("\n\n".join((to_compare['headline'] + ' ' + to_compare['item']).values.tolist()), return_tensors="pt")


import torch
from tqdm import tqdm

nlls = []
cum_ix = 0
for i, row in to_compare.iterrows():

    encodings_1 = tokenizer(row['headline'], return_tensors="pt")
    encodings_2 = tokenizer(row['item'], return_tensors="pt")

    trg_len = encodings_2.input_ids.shape[1]
    input_ids = torch.cat((encodings_1.input_ids[:, :], encodings_2.input_ids[:, :]), dim=1)  # .to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over input tokens.
        # Multiply it with trg_len to get the summation instead of average.
        # We will take average over all the tokens to get the true average
        # in the last step of this example.
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)


# ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
