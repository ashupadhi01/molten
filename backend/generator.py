import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from models import GenerationConfig
from utils import get_memory_usage

print(torch.__config__.parallel_info())
# torch.set_num_threads(1)


class CustomGenerator():
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache_size: float = None

    @torch.inference_mode
    def generate(
        self,
        text: str,
        generation_config: GenerationConfig,
    ):
        inputs = self.tokenizer(text, return_tensors = "pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None
        total_kv_size = 0

        for _ in range(generation_config.max_new_tokens):

            output = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                use_cache = generation_config.use_cache,
                past_key_values = past_key_values
            )

            token_id = self._sample(output.logits[0][-1])

            if self._should_stop(token_id.item()):
                break

            token = self.tokenizer.decode(token_id)

            if generation_config.use_cache:
                input_ids = token_id.unsqueeze(0).unsqueeze(0)
                past_key_values = output.past_key_values
                # total_kv_size += self._compute_kv_cache_size(past_key_values)
                attention_mask = self._update_attention_mask(input_ids.shape[0], past_key_values.get_seq_length() + 1)

            else:
                input_ids = self._append_token(input_ids, token_id)
                attention_mask = self._update_attention_mask(input_ids.shape[0], input_ids.shape[1])

            yield token

        print(f"Cumulative sum of all KV cache tensor for the generation: {total_kv_size}")

    def count_prompt_tokens(self, text: str):
        return len(self.tokenizer(text)["input_ids"])

    def _should_stop(self, token_id: int):
        return self.tokenizer.eos_token_id == token_id

    def _append_token(self, input_ids: torch.Tensor, token_id: torch.Tensor):
        return torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim = 1)

    def _update_attention_mask(self, batch_dim: int, seq_len: int):
        return torch.ones(batch_dim, seq_len)

    def _sample(self, logits: torch.Tensor):
        return logits.argmax()


if __name__ == "__main__":
    import os
    import sys
    resource_path = os.path.join(os.path.expanduser('~'), 'models/SmolLM2-360M')
    # resource_path = os.path.join(os.path.expanduser('~'), 'models/gpt2')

    print(resource_path)
    print(f"Process ID: {os.getpid()}")


    # Initialise the model
    model = AutoModelForCausalLM.from_pretrained(
        resource_path,
        dtype = torch.float16,
    )

    # Initialise the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        resource_path
    )

    generator = CustomGenerator(model = model, tokenizer = tokenizer)
    
    def test(use_cache: bool):
        print("\nCACHE USE: ", use_cache)

        config = GenerationConfig(use_cache = use_cache)

        while True:
            output_seq_len = int(input("OUTPUT_SEQ_LEN: "))
            text = input("INPUT: ")

            k = generator.tokenizer(text)["input_ids"]
            print(f"\nINPUT_SEQUENCE: {k}\n")
            config.max_new_tokens = output_seq_len

            curr_footprint = get_memory_usage() / 1024
            generator.generate(text, config)
            after_footprint = get_memory_usage() / 1024

            print(f"Process footprint before generation: {curr_footprint}")
            print(f"Process footprint after generation: {after_footprint}")
            print(f"Generation process memory diff: {after_footprint - curr_footprint} MB\n\n")
            
    test(use_cache = False if sys.argv[1] == "false" else True)
    # uv run generator.py true
