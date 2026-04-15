import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

import argparse

def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tensor_parallel_size, kv_quant=args.kv_quant)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
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
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

# def main(args):
#     path = os.path.expanduser(args.model_path)
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

#     sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
#     # 直接写成问答的形式，引导基座模型接龙
#     raw_prompts = [
#         "Question: Please introduce yourself.\nAnswer:",
#         "Question: List all prime numbers within 100.\nAnswer:",
#     ]
    
#     # 删掉 tokenizer.apply_chat_template 那一大段，直接传 raw_prompts
#     outputs = llm.generate(raw_prompts, sampling_params)

#     for prompt, output in zip(raw_prompts, outputs):
#         print("\n" + "="*40)
#         print(f"Prompt: {prompt!r}")
#         print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano vllm")
    parser.add_argument("--model_path", type=str, default="~/huggingface/Qwen3-0.6B/", help="model path")
    parser.add_argument("--enforce_eager", action="store_true", help="enforce eager mode")
    parser.add_argument("--tensor_parallel_size", "--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--temperature", type=float, default=0.6, help="sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens to generate")
    parser.add_argument("--kv_quant", action="store_true", help="use quantization for kv cache")
    args = parser.parse_args()
    main(args)
