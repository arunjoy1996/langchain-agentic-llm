# agent.py

import os
import asyncio
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables import ConfigurableField
from langchain.callbacks.base import AsyncCallbackHandler

from langchain_groq import ChatGroq
from tools import TOOLS
# agent.py


# === ENV VARS ===
os.environ["GROQ_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "CoT agentic LLM"

# === Prompt ===
cot_system_prompt = """
You are a helpful and methodical assistant. Your goal is to answer the user's question through clear reasoning and effective use of tools.

Follow this approach:

1. If the question is simple and you can answer it confidently from your own knowledge, do so directly without using tools.

2. If the question is complex but solvable without tools:
   - Break it into smaller subproblems and state them.
   - Solve each step-by-step with clear reasoning.

3. If tools are required:
   - Use them purposefully and only when needed.
   - Do not repeat tool usage unnecessarily.

4. Always conclude with the `final_answer` tool, summarizing the result clearly for the user.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", cot_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# create tool name to function mapping
name2tool = {tool.name: tool.func for tool in TOOLS}
# === LLM Setup ===
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.0, streaming=True)

llm = llm.configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)

class QueueCallbackHandler(AsyncCallbackHandler):
    """Callback handler that puts tokens into a queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False
        self.cot_ended = False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()

            if token_or_done == "<<DONE>>":
                # this means we're done
                return
            if token_or_done:
                yield token_or_done

    async def on_tool_start(self, *args, **kwargs):
        tool = kwargs.get("name", "")
        print(f"🔧 Tool starting: {tool}")
        # Optionally send a message to the queue
        self.queue.put_nowait(f"<<TOOL_START:{tool}>>")

    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        """Put new token in the queue."""
        chunk = kwargs.get("chunk")
        if chunk:
            # print(chunk)
            # check for final_answer tool call
            if tool_calls := chunk.message.additional_kwargs.get("tool_calls"):
                if tool_calls and not self.cot_ended:
                    self.queue.put_nowait("<<COT_ENDED>>")
                    self.cot_ended = True
                if tool_calls[0]["function"]["name"] == "final_answer":
                    # this will allow the stream to end on the next `on_llm_end` call
                    self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
        return
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        """Put None in the queue to signal completion."""
        # tool call
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")
        return
    

from langchain_core.messages import ToolMessage

class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, max_iterations: int = 3):
        self.chat_history = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(TOOLS)  # we're forcing tool use again
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad = []
        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            async def stream(query: str):
                response = self.agent.with_config(
                    callbacks=[streamer]
                )
                # we initialize the output dictionary that we will be populating with
                # our streamed output
                output = None
                # now we begin streaming
                async for token in response.astream({
                    "input": query,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad
                }):
                    if output is None:
                        output = token
                    else:
                        # we can just add the tokens together as they are streamed and
                        # we'll have the full response object at the end
                        output += token
                    if token.content != "":
                        # we can capture various parts of the response object
                        if verbose: print(f"content: {token.content}", flush=True)
                    tool_calls = token.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        if verbose: print(f"tool_calls: {tool_calls}", flush=True)
                        tool_name = tool_calls[0]["function"]["name"]
                        if tool_name:
                            if verbose: print(f"tool_name: {tool_name}", flush=True)
                        arg = tool_calls[0]["function"]["arguments"]
                        if arg != "":
                            if verbose: print(f"arg: {arg}", flush=True)
                return AIMessage(
                    content=output.content,
                    tool_calls=output.tool_calls,
                    tool_call_id=output.tool_calls[0]["id"]
                )

            tool_call = await stream(query=input)
            # add initial tool call to scratchpad
            agent_scratchpad.append(tool_call)
            # otherwise we execute the tool and add it's output to the agent scratchpad
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_call_id
            tool_out = name2tool[tool_name](**tool_args)
            # add the tool output to the agent scratchpad
            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            agent_scratchpad.append(tool_exec)
            count += 1
            # if the tool call is the final answer tool, we stop
            if tool_name == "final_answer":
                break
        # add the final output to the chat history, we only add the "answer" field
        final_answer = tool_out["answer"]
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])
        # return the final answer in dict form
        return tool_args

agent_executor = CustomAgentExecutor() 


# For import in main.py
agent_executor = CustomAgentExecutor()
