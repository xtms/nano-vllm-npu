import argparse
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_device_id(s: str) -> int | list[int]:
    """Parse --device-id: comma-separated → list[int], else int."""
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return int(s)


def main():
    parser = argparse.ArgumentParser(description="nano-vllm NPU inference example")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--device-type", type=str, default="npu",
                        choices=["cuda", "npu"],
                        help="Device type (default: npu)")
    parser.add_argument("--memory-utilization", type=float, default=0.9,
                        help="Fraction of device memory for KV cache (default: 0.9)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallelism degree (default: 1)")
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable graph capture")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum number of tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length (default: 4096)")
    parser.add_argument("--device-id", type=parse_device_id, default=0,
                        help="Device ID(s) to use (default: 0). "
                             "Single int: offset, e.g. --device-id 2 with TP=2 uses cards 2,3. "
                             "Comma-separated list: explicit mapping, e.g. --device-id 2,4,6 with TP=3")

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    print(f"  device_type:         {args.device_type}")
    print(f"  device_id:           {args.device_id}")
    print(f"  memory_utilization:  {args.memory_utilization}")
    print(f"  tensor_parallel_size:{args.tensor_parallel_size}")
    print(f"  enforce_eager:       {args.enforce_eager}")
    print(f"  max_model_len:       {args.max_model_len}")

    llm = LLM(
        args.model,
        device_type=args.device_type,
        device_id=args.device_id,
        memory_utilization=args.memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print()
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
