from typing import List
import json
from json import JSONEncoder

# -----------------lora数据集的结构--------------------------
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class Conversation:
    def __init__(self, conversations: List[Message]):
        self.conversations = conversations

# -----------------git commit数据集的结构--------------------------
# diff_id: 对应 diff_id 字段，类型为整数。
# repo: 对应 repo 字段，表示仓库路径（格式为 "owner/repo"）。
# sha: 对应 sha 字段，是提交的哈希值。
# time: 对应 time 字段，表示时间戳（ISO 8601 格式）。
# diff: 对应 diff 字段，包含代码变更内容。
# msg: 对应 msg 字段，表示提交信息。
class DiffData:
    def __init__(
        self,
        diff_id: int,
        repo: str,
        sha: str,
        time: str,
        diff: str,
        msg: str
    ):
        self.diff_id = diff_id
        self.repo = repo
        self.sha = sha
        self.time = time
        self.diff = diff
        self.msg = msg

class MyJSONEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

if __name__ == "__main__":
    file_path = "../dataset/py.jsonl"
    output_path = "../dataset/lora_python_commit.jsonl"
    with open(file_path, 'r', encoding='utf-8') as f:
        with open(output_path, 'a', encoding='utf-8') as fw:
            for line in f:
                # 解析 JSON 到对象
                data = json.loads(line)
                diff_data = DiffData(**data)
                # 创建Message问答对
                prompt: str = f"""你是一个git提交信息生成模型，你需要根据我提供的git差异内容生成提交信息：```{diff_data.diff}```"""
                message1 = Message("user", prompt)
                message2 = Message("assistant", diff_data.msg)
                conversation = Conversation([message1, message2])
                # 将对话转换为JSON
                json_str = json.dumps(conversation, cls=MyJSONEncoder, ensure_ascii=False)
                # 写入文件
                fw.write(json_str + '\n')