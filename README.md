# Hugging Face VSCode Endpoint Server

starcoder server for [huggingface-vscdoe](https://github.com/huggingface/huggingface-vscode) custom endpoint.

**Can't handle distributed inference very well yet.**

## Fork

This fork:

- Refactor the generator codes to separate classes
- Adds support for starcoder under ct2fast conversion for faster inference on consumer hardware
- Has a support vs code extension for triggered code completion see [vstarcoder](https://github.com/piratos/vstarcoder)

PS: Rationale for not using huggingface-vscode explained in vstarcoder extension readme

## Usage

```shell
pip install -r requirements.txt
python main.py
```

Fill `http://localhost:8000/api/generate/` into `Hugging Face Code > Model ID or Endpoint` in VSCode.

## API

```shell
curl -X POST http://localhost:8000/api/generate/ -d '{"inputs": "", "parameters": {"max_new_tokens": 64}}'
# response = {"generated_text": ""}
```
