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

            # add the new token_id to input sequence and extend the attention mask
            inputs["input_ids"] = torch.cat([inputs["input_ids"], token_id.unsqueeze(0).unsqueeze(0)], dim = 1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.tensor([[1]])], dim = 1)

            print(token, end = "", flush = True)

    def _should_stop(self, token_id: int):
        return self.tokenizer.eos_token_id == token_id

    def _append_token(self, inputs: BatchEncoding, token_id: torch.Tensor):
        pass





if __name__ == "__main__":
    import os
    resource_path = os.path.join(os.path.expanduser('~'), 'models/gpt2')
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

    """
    GPT2Tokenizer(
        name_or_path='/home/ashutosh/models/gpt2',
        vocab_size=50257,
        model_max_length=1024,
        padding_side='right',
        truncation_side='right',
        special_tokens = {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'},
        added_tokens_decoder={50256: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True)}
    )
    """

    # # print(model.config)
    # generator = CustomGenerator(model = model, tokenizer = tokenizer)
    # config = SamplingConfig()

    text = "How are you?"
    k = tokenizer(text, return_tensors = 'pt')
    print(k, type(k))
    # generator.generate(text, config)