#!/usr/bin/env python3
import argparse
import random
import sys
import subprocess

import torch
from colorama import init, Fore, Style
from transformers import TextStreamer

from eval_model import setup_seed, init_model

init(autoreset=True)  # 初始化 colorama 以支持 Windows 和 ANSI 颜色


def print_help(msg=None):
    """打印帮助信息，带颜色"""
    if msg:
        print(f"{Fore.RED}{msg}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Usage:{Style.RESET_ALL}")
    print(f"  python {sys.argv[0]} 'git diff --cached'")


def main():
    args = argparse.Namespace(
        lora_name='None',
        out_dir='out',
        temperature=0.85,
        top_p=0.85,
        device='cuda' if torch.cuda.is_available()  else 'mps' if torch.backends.mps.is_available() else 'cpu',
        hidden_size=512,
        num_hidden_layers=8,
        max_seq_len=8192,
        use_moe=False,
        history_cnt=0,
        load=1,
        model_mode=2
    )
    # print('[DEBUG] args: ', args)
    # print(f"[DEBUG] sys.argv: {sys.argv}")
    print("[DEBUG] 目前使用的推理设备为", args.device)


    # 读取命令行参数
    cmd_arg = None
    if len(sys.argv) > 1:
        cmd_arg = sys.argv[1]
        # print("[DEBUG] cmd_arg: ", cmd_arg)

    # 初始化模型和tokenizer
    model, tokenizer = init_model(args)
    setup_seed(random.randint(0, 2048))

    # 获得流式输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 获取输入内容
    user_prompt = input("👶: ")
    input_data = ""
    if cmd_arg:
        # 执行命令获取 diff 内容
        result = subprocess.run(cmd_arg, shell=True, capture_output=True, text=True)
        input_data = result.stdout
        # print("[DEBUG] input_data: ", input_data)
    else:
        print_help("No git diff or text was passed to this function.")
        return 1

    # 构造 prompt
    # 提示词工程
    prompt = f"""
    你是一个git提交信息生成模型，你需要根据我提供的git差异内容生成3条提交信息。分点，用中文回答。例如：
    1. 添加了何种功能
    2. 修复了何种bug
    3. 优化了何种性能
    
    请根据以下给定的git差异内容生成3条提交信息，可以参考用户的提示\"{user_prompt}\"：
    ```{input_data}```
    """
    # print('[DEBUG] prompt: ', prompt)

    # 调用minimind本地模型
    try:
        # 构造messages
        messages = [{"role": "user", "content": prompt}]

        # 将用户与助手之间的多轮对话转换为模型可以理解的格式(一种特殊的格式，类似于json)
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        # # 将文本输入 new_prompt 转换为模型可以处理的张量格式（token IDs 和 attention mask）
        # inputs = tokenizer(
        #     new_prompt,
        #     return_tensors="pt",
        #     truncation=True
        # ).to(args.device)

        try:
            # 将文本输入 new_prompt 转换为模型可以处理的张量格式（token IDs 和 attention mask）
            inputs = tokenizer(
                new_prompt,
                return_tensors="pt",
                truncation=True
            ).to(args.device)
        except Exception as e:
            if 'MPS' in str(e):
                print("检测到MPS错误，自动回退到CPU模式")
                # 将文本输入 new_prompt 转换为模型可以处理的张量格式（token IDs 和 attention mask）
                inputs = tokenizer(
                    new_prompt,
                    return_tensors="pt",
                    truncation=True
                ).to("cpu")
            else:
                raise e

        print('🤖️: ', end='')
        # 得到模型输出的tokenIds
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature
        )
        # print('[DEBUG] generated_ids: ', generated_ids)

        # 根据模型生成的tokenIds，生成对应的response(文本)
        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')
        # todo: 采用git comment数据集进行微调（数据来源：实验室内部已有的git comment，通过脚本组装成数据集）
        # todo: 可以改进的点：将上一版完整代码进行RAG后召回相关代码作为上下文，提升模型的理解能力
        return None
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    main()