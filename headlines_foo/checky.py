import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer("Today is a nice day", return_tensors="pt").input_ids

generated_outputs = gpt2.generate(input_ids, do_sample=True, num_return_sequences=3, output_scores=True)

# only use id's that were generated
# gen_sequences has shape [3, 15]
gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

# let's stack the logits generated at each step to a tensor and transform
# logits to probs
probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]
