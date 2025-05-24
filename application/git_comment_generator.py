#!/usr/bin/env python3
import sys
import subprocess
from colorama import init, Fore, Style

init(autoreset=True)  # 初始化 colorama 以支持 Windows 和 ANSI 颜色

def print_help(msg=None):
    """打印帮助信息，带颜色"""
    if msg:
        print(f"{Fore.RED}{msg}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Usage:{Style.RESET_ALL}")
    print(f"  git diff --cached | {sys.argv[0]}")
    print(f"{Fore.LIGHTBLACK_EX}  # or{Style.RESET_ALL}")
    print(f"  {sys.argv[0]} 'git diff --cached'")


def main():
    # 读取管道输入
    pipe_input = ""
    if not sys.stdin.isatty():
        pipe_input = sys.stdin.read()

    # 读取命令行参数
    cmd_arg = None
    if len(sys.argv) > 1:
        cmd_arg = sys.argv[1]

    # 检查输入冲突
    if cmd_arg and pipe_input:
        print_help("Both a piped value and a function arg were passed to this function, but only one can be accepted.")
        return 1

    # 获取输入内容
    input_data = ""
    if cmd_arg:
        # 执行命令获取 diff 内容
        result = subprocess.run(cmd_arg, shell=True, capture_output=True, text=True)
        input_data = result.stdout
    elif pipe_input:
        input_data = pipe_input
    else:
        print_help("No git diff or text was passed to this function.")
        return 1

    # 构造 prompt
    # todo: 提示词工程
    codeblocked_git_diff = f"```{input_data}```\nCould you please generate 3 commit messages (preferably with a body going into more detail about the commit) for the git diff I've just given you?"

    # 调用minimind本地模型
    try:
        # todo: 调用minimind模型生成comment
        # todo: 采用git comment数据集进行微调（数据来源：实验室内部已有的git comment，通过脚本组装成数据集）
        # todo: 可以改进的点：将上一版完整代码进行RAG后召回相关代码作为上下文，提升模型的理解能力



if __name__ == "__main__":
    main()