# main.py
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.staticfiles import StaticFiles
from agent import agent_executor, QueueCallbackHandler

import os

STATIC_DIR = os.path.join(os.path.dirname(__file__), "generated_images")



app = FastAPI()
app.mount("/generated_images", StaticFiles(directory=STATIC_DIR), name="generated_images")

@app.get("/stream")
async def stream_response(request: Request, query: str):
    queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)

    async def event_generator():
        # Launch the agent in the background
        agent_task = asyncio.create_task(agent_executor.invoke(query, streamer))
        cot_ended = False
        while True:
            if await request.is_disconnected():
                agent_task.cancel()
                break

            token = await queue.get()
            if token == "<<DONE>>":
                print("found done token")
                break
            if token=="<<COT_ENDED>>":
                cot_ended= True
                yield f"data: \n\n\n\n"

            elif cot_ended== False:               
                yield f"data: {token.message.content} \n\n"
                           
            elif token != "<<STEP_END>>":
                token= token.message
                
                if token.tool_calls and token.tool_calls[0]["name"] == "final_answer":
                     answer = token.tool_calls[0]["args"]["answer"]
                     yield f"data: {answer}\n\n"

                  


    return StreamingResponse(event_generator(), media_type="text/event-stream")
