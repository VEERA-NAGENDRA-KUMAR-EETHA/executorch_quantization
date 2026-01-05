import os
import time
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from executorch.runtime import Runtime

# --- Sampling Function ---
def top_p_sampling(probs, p=0.9, seed=200):
    sorted_probs, sorted_inds = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff_index = torch.sum(cum_probs <= p).item()
    cutoff_index = max(cutoff_index, 1)

    sorted_probs[cutoff_index:] = 0
    sorted_probs /= sorted_probs.sum()
    torch.manual_seed(seed)
    ind = torch.multinomial(sorted_probs, num_samples=1).squeeze().item()
    return sorted_inds[ind]


question = "How does the mobile app connect to the refrigerator?"
prompt = f"###{question} ###Long Answer: "

stokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
runtime = Runtime.get()
# program = runtime.load_program("new_rag_llm.pte")  # model file for fp32
program = runtime.load_program("new_rag_llm_int8.pte")  # model file for int8
method = program.load_method("forward")

# Hidden states MUST be float32 for the XNNPACK interface you exported
h = torch.zeros(24, 4096, dtype=torch.float32)
fifo = torch.zeros(24, 4096, 4, dtype=torch.float32)

# Tokenize prompt
input_ids = stokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]

output_ids = []
full_stop_count = 0
n_fullstops = 2
max_new_tokens = 80

# --- Feed prompt (Prefill) ---
print("Processing prompt...")
ts_start = time.time()
for tkn_i in input_ids:
    # IMPORTANT: Use .view(1) to match the 1D [1] shape from your export
    # .unsqueeze(0) would create [1, 1] which causes the 0x1 error.
    current_input = tkn_i.view(1).to(torch.long)
    print(current_input.shape)
    
    out = method.execute((current_input, h, fifo))
    
    logits = out[2]
    # Use clone() to ensure we have a fresh copy for the next iteration
    h = out[3].clone().detach()
    fifo = out[4].clone().detach()

ttft = time.time() - ts_start

# --- Start autoregressive generation ---
# Sample the first token from the prompt's final logits
probs = F.softmax(logits.squeeze(0), dim=-1)
input_id = top_p_sampling(probs)
output_ids.append(input_id.item())

print(f"First token: {input_id.item()} | Decoded: '{stokenizer.decode([input_id.item()])}'")

ts_gen_start = time.time()

for step in range(max_new_tokens):
    # Maintain 1D shape [1]
    current_input = input_id.view(1).to(torch.long)
    
    out = method.execute((current_input, h, fifo))
    
    logits = out[2]
    h = out[3].clone().detach()
    fifo = out[4].clone().detach()

    probs = F.softmax(logits.squeeze(0), dim=-1)
    next_token = top_p_sampling(probs)
    output_ids.append(next_token.item())
    
    input_id = next_token

    # Check for stop conditions (Full stop or EOS)
    if next_token.item() == 28723: # '.'
        full_stop_count += 1
    if next_token.item() == 2 or full_stop_count >= n_fullstops:
        break

ts_end = time.time()
t_infer = ts_end - ts_gen_start

# --- Output Results ---
print("\n" + "="*30)
print(f"TTFT (Prefill): {ttft:.2f}s")
print(f"Generation: {t_infer:.2f}s ({len(output_ids)/t_infer:.2f} tokens/sec)")
print(f"Total Tokens: {len(output_ids)}")
print("="*30)
print("Decoded output:\n", stokenizer.decode(output_ids))
