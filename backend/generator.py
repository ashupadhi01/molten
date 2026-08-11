import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from models import SamplingConfig

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
        sampling_config: SamplingConfig
    ):
        inputs = self.tokenizer(text, return_tensors = "pt")

        for _ in range(sampling_config.max_new_tokens):
            output = self.model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"]
            )

            token_id = output.logits[0][-1].argmax()

            if self._should_stop(token_id):
                break

            token = tokenizer.decode(token_id)
            inputs = self._append_token(inputs, token_id)

            print(token, end = "", flush = True)

    def generate_with_cache(
        self,
        text: str,
        sampling_config: SamplingConfig
    ):
        inputs = self.tokenizer(text, return_tensors = "pt")

        # Prefill
        output = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"]
        )

        kv_cache = output.past_key_values
        attention_mask = torch.ones(1, kv_cache.get_seq_length() + 1)

        token_id = self._sample(output.logits[0][-1])
        token = self.tokenizer.decode(token_id)

        print(token, end = "", flush = True)

        # Decode
        for _ in range(sampling_config.max_new_tokens - 1):
            output = self.model(
                input_ids = token_id.unsqueeze(0).unsqueeze(0),
                attention_mask = attention_mask,
                past_key_values = kv_cache
            )

            kv_cache = output.past_key_values
            print("KV cache length after decode", kv_cache.get_seq_length())

            token_id = self._sample(output.logits[0][-1])

            if self._should_stop(token_id):
                break

            # Update attention mask
            attention_mask = torch.ones(1, kv_cache.get_seq_length() + 1)
            token = self.tokenizer.decode(token_id)

            print(token, end = "", flush = True)

    def _should_stop(self, token_id: int):
        return self.tokenizer.eos_token_id == token_id

    def _append_token(self, inputs: BatchEncoding, token_id: torch.Tensor):
        inputs["input_ids"] = torch.cat([inputs["input_ids"], token_id.unsqueeze(0).unsqueeze(0)], dim = 1)
        inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape[0], inputs["input_ids"].shape[1])
        return inputs

    def _sample(self, logits: torch.Tensor):
        return logits.argmax()


if __name__ == "__main__":
    import os
    resource_path = os.path.join(os.path.expanduser('~'), 'models/SmolLM2-360M-Instruct')
    # resource_path = os.path.join(os.path.expanduser('~'), 'models/gpt2')

    print(resource_path)


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
    config = SamplingConfig(max_new_tokens = 1000)

    text = "Write an essay on India?"
    k = tokenizer(text, return_tensors = 'pt')
    print("Input Sequence: ", k["input_ids"])

    generator.generate_with_cache(text, config)
    # generator.generate(text, config)
