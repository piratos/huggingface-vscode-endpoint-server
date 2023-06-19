import asyncio

from hf_hub_ctranslate2 import GeneratorCT2fromHfHub

from generators import GeneratorBase


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
            end_token=["<|endoftext|>", "ĊĊ", "@", "class", "def"],
            callback=_callback if asynchronous else callback
        )
        return outputs
