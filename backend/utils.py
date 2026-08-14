import os
import torch
from transformers import AutoTokenizer


def print_token_logits(
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    num_tokens: int
):
    print("-" * 50)
    for i in range(num_tokens):
        print(f"{tokenizer.convert_ids_to_tokens(input_ids[0][i].unsqueeze(0))[0]}: {logits[0][i][:3].tolist()}...")
    print("-" * 50)


def get_memory_usage():
    with open(f"/proc/{os.getpid()}/smaps_rollup") as f:
        uss_kb = sum(
            int(line.split()[1])
            for line in f
            if line.startswith(("Private_Clean", "Private_Dirty"))
        )
    return uss_kb