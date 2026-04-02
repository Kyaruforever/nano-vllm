import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

import argparse

def main(model_path: str):
    path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
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

# def main(model_path: str):
#     path = os.path.expanduser(model_path)
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
    args = parser.parse_args()
    main(args.model_path)
