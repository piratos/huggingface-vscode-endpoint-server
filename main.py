import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from generators import get_model, GeneratorBase
import json

from util import logger, get_parser

app = FastAPI()
app.add_middleware(
    CORSMiddleware
)
generator: GeneratorBase = ...


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    is_hf = False
    is_oa = False
    max_tokens: int = 0
    prompt: str = ""
    if 'inputs' in json_request:
        is_hf = True
        prompt = json_request['inputs']
        max_tokens = json_request['parameters']['max_new_tokens']
    elif 'prompt' in json_request:
        is_oa = True
        prompt = json_request['prompt']
        max_tokens = json_request['max_tokens']
    parameters: dict = {'max_new_tokens': max_tokens}
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(prompt)}')
    generated_text: str = generator.generate(prompt, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    if is_oa:
        return {
            "choices": [
                {"text": generated_text}
            ],
            "status": 200
        }
    elif is_hf:
        return {
            "generated_text": generated_text,
            "status": 200
        }
    else:
        return None

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection
    parameters = {}
    await websocket.accept()
    async def reply_word(step):
        word = step.token.replace('Ġ', ' ').replace('Ċ', '\n')
        await websocket.send_text(word)
    try:
        # Receive the line from the client
        line = await websocket.receive_text()
        # Generate the text 
        generated_text = generator.generate(line, parameters, callback=reply_word, asynchronous=True)
        # Send the generated text back to the client
        #await websocket.send_text(generated_text)
        await websocket.close()
    except Exception as e:
        print(f"WebSocket error: {e}")



def main():
    global generator
    args = get_parser().parse_args()
    generator = get_model(args.pretrained)(args.pretrained, device_map='auto')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
