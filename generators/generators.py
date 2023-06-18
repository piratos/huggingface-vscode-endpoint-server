from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from transformers import Pipeline, pipeline
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
from transformers import AutoTokenizer
import asyncio

HAS_PYTORCH = False
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    pass


class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)


class StarCoder(GeneratorBase):
    def __init__(self, pretrained: str, device: str = None, device_map: str = None):
        self.pretrained: str = pretrained
        self.pipe: Pipeline = pipeline(
            "text-generation", model=pretrained, torch_dtype=torch.bfloat16, device=device, device_map=device_map)
        self.generation_config = GenerationConfig.from_pretrained(pretrained)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        return generated_text


class SantaCoder(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.model.to(device=self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.generation_config: GenerationConfig = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        output_ids: torch.Tensor = self.model.generate(input_ids, generation_config=config)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text


class ReplitCode(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.default_parameter: dict = dict(
            do_sample=True, top_p=0.95, top_k=4, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, query: str, parameters: dict = None) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        params = {**self.default_parameter, **(parameters or {})}
        params.pop('stop')
        output_ids: torch.Tensor = self.model.generate(input_ids, **params)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

class StarCoderCT2Fast(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda', device_map: str = None):
        # Name or path to a ct2fast model
        self.pretrained: str = pretrained
        self.device: str = device
        self.max_tokens: int = 512
        self.model: GeneratorCT2fromHfHub = GeneratorCT2fromHfHub(
            model_name_or_path=pretrained,
            device=self.device,
            compute_type="int8_float16"
        )
    
    def clean_token(self, token):
        return string.replace('Ġ', ' ').replace('Ċ', '\n')
    def print_token(self, token):
        print(token)

    def generate(self, query: str, parameters: dict = None, callback = None, asynchronous = False) -> str:
        max_tokens = max(parameters.get("max_new_tokens", 0), self.max_tokens)
        output_text = ""
        def _callback(step):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(callback(step))
            loop.close()
        outputs = self.model.generate(
            text=query,
            max_length=max_tokens,
            include_prompt_in_result=False,
            end_token=["<|endoftext|>", "ĊĊ"],
            callback=_callback if asynchronous else callback
        )
        return outputs
