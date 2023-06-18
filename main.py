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
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')
    generated_text: str = generator.generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "generated_text": generated_text,
        "status": 200
    }

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
