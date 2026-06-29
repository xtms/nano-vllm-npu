import argparse
import asyncio
import json
import time
import uuid
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanovllm import LLM, SamplingParams


# ---------- Request / Response models ----------

class SamplingParamsRequest(BaseModel):
    temperature: float = 0.6
    max_tokens: int = 256
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    prompt: Union[str, list[int]]
    sampling_params: SamplingParamsRequest = Field(default_factory=SamplingParamsRequest)


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    sampling_params: SamplingParamsRequest = Field(default_factory=SamplingParamsRequest)


# ---------- API Server ----------

def create_app(args):
    app = FastAPI(title="nano-vllm", version="0.2.0")

    print(f"Loading model from {args.model}...")
    print(f"  device_type:         {args.device_type}")
    print(f"  device_id:           {args.device_id}")
    print(f"  memory_utilization:  {args.memory_utilization}")
    print(f"  tensor_parallel_size:{args.tensor_parallel_size}")
    print(f"  enforce_eager:       {args.enforce_eager}")
    print(f"  max_model_len:       {args.max_model_len}")
    print(f"  max_num_seqs:        {args.max_num_seqs}")

    llm = LLM(
        args.model,
        device_type=args.device_type,
        device_id=args.device_id,
        memory_utilization=args.memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    print("Model loaded successfully!")

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_name = args.served_model_name or args.model

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nano-vllm",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        request_id = str(uuid.uuid4())
        sp = SamplingParams(
            temperature=request.sampling_params.temperature,
            max_tokens=request.sampling_params.max_tokens,
            ignore_eos=request.sampling_params.ignore_eos,
        )

        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = tokenizer.encode(prompt)

        outputs = llm.generate([prompt], sp)
        output = outputs[0]

        return JSONResponse(content={
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": output["text"],
                    "token_ids": output["token_ids"],
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(output["token_ids"]),
                "total_tokens": len(prompt) + len(output["token_ids"]),
            },
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        request_id = str(uuid.uuid4())
        sp = SamplingParams(
            temperature=request.sampling_params.temperature,
            max_tokens=request.sampling_params.max_tokens,
            ignore_eos=request.sampling_params.ignore_eos,
        )

        prompt = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_ids = tokenizer.encode(prompt)
        outputs = llm.generate([prompt_ids], sp)
        output = outputs[0]

        return JSONResponse(content={
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(output["token_ids"]),
                "total_tokens": len(prompt_ids) + len(output["token_ids"]),
            },
        })

    return app


def parse_device_id(s: str) -> int | list[int]:
    """Parse --device-id: comma-separated → list[int], else int."""
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return int(s)


def main():
    parser = argparse.ArgumentParser(description="nano-vllm API Server")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--device-type", type=str, default="npu",
                        choices=["cuda", "npu"],
                        help="Device type (default: npu)")
    parser.add_argument("--memory-utilization", type=float, default=0.9,
                        help="Fraction of device memory for KV cache (default: 0.9)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallelism degree (default: 1)")
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable graph capture")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length (default: 4096)")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Maximum number of concurrent sequences (default: 256)")
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="Model name for API (default: model path)")
    parser.add_argument("--device-id", type=parse_device_id, default=0,
                        help="Device ID(s) to use (default: 0). "
                             "Single int: offset, e.g. --device-id 2 with TP=2 uses cards 2,3. "
                             "Comma-separated list: explicit mapping, e.g. --device-id 2,4,6 with TP=3")

    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
