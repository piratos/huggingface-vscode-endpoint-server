import os

from transformers import PretrainedConfig

from generators.base import GeneratorBase
from generators.santacoder import SantaCoder
from generators.starcoder import StarCoder
from generators.replit import ReplitCode
from generators.starcoderct2fast import StarCoderCT2Fast

def get_model(pretrained_or_path):
    if os.path.exists(pretrained_or_path):
        model_path = pretrained_or_path
        # /!\ trust remote code is dangerous
        config_dict, _ = PretrainedConfig.get_config_dict(
            model_path, trust_remote_code=True
        )
        model_type = config_dict["model_type"]

        if model_type == "gpt_bigcode_ct2fast":
            return StarCoderCT2Fast
        else:
            raise NotImplementedError("only ct2fast starcoder are supported now")

    else:
        model_name = pretrained_or_path.lower()
        if "starcoder" in model_name:
            return StarCoder
        elif "santacoder" in model_name:
            return SantaCoder
        elif "replit" in model_name:
            return ReplitCode
        elif "ct2fast" in model_name:
            return StarCoderCT2Fast


__all__ = [
    "GeneratorBase",
    "SantaCoder",
    "StarCoder",
    "ReplitCode",
    "StarCoderCT2Fast",
    "get_model",

]
