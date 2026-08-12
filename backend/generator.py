import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from models import GenerationConfig

class CustomGenerator():
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        text: str,
        generation_config: GenerationConfig
    ):
        inputs = self.tokenizer(text, return_tensors = "pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None

        for _ in range(generation_config.max_new_tokens):

            output = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                use_cache = generation_config.use_cache,
                past_key_values = past_key_values
            )

            token_id = output.logits[0][-1].argmax()

            # if self._should_stop(token_id.item()):
                # break

            token = self.tokenizer.decode(token_id)

            if generation_config.use_cache:
                input_ids = token_id.unsqueeze(0).unsqueeze(0)
                past_key_values = output.past_key_values
                self._compute_kv_cache_size(past_key_values)
                attention_mask = self._update_attention_mask(input_ids.shape[0], past_key_values.get_seq_length() + 1)

            else:
                input_ids = self._append_token(input_ids, token_id)
                attention_mask = self._update_attention_mask(input_ids.shape[0], input_ids.shape[1])

            # print(token, end = "", flush = True)
            # yield token

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

    def _compute_kv_cache_size(self, past_key_values: DynamicCache):
        total_bytes = 0

        for key, value, _ in past_key_values:
            total_bytes += (key.element_size() * key.numel() + value.element_size() * value.numel())

        print(f"Seq length: {past_key_values.get_seq_length()}, KV cache: {total_bytes / 1024:.2f} KB, i.e., {(total_bytes / (1024 * 1024)):.2f} MB")

if __name__ == "__main__":
    import os
    import sys
    # resource_path = os.path.join(os.path.expanduser('~'), 'models/SmolLM2-360M-Instruct')
    resource_path = os.path.join(os.path.expanduser('~'), 'models/gpt2')

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
    # text = "What"

    # print(generator.count_prompt_tokens(text))
    
    def test(max_new_tokens: int, use_cache: bool):
        print("CACHE USE: ", use_cache)

        config = GenerationConfig(
            max_new_tokens = max_new_tokens,
            use_cache = use_cache
        )

        text = "apple"
        k = generator.tokenizer(text)["input_ids"]

        print(f"INPUT SEQUENCE: {k}\n")

        generator.generate(text, config)

    test(max_new_tokens = int(sys.argv[1]), use_cache = False if sys.argv[2] == "false" else True)
    # uv run generator.py 5 true :example command to run a sample with KV cache
