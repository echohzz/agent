"""Tool definitions, registration data, and argument validation."""

import ast
import math
import operator
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: object


def calculate(args, _user, _session):
    expression = args["expression"]
    if len(expression) > 100:
        raise ValueError("表达式过长")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node, depth=0):
        if depth > 16:
            raise ValueError("表达式嵌套过深")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("指数过大")
            value = binary[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            value = unary[type(node.op)](evaluate(node.operand, depth + 1))
        else:
            raise ValueError("只支持数字、括号和 + - * / **")
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("计算结果超出范围")
        return value

    return {"result": evaluate(ast.parse(expression, mode="eval").body)}


SEARCH_DOCS = [
    {"title": "Agent 循环", "keywords": ("agent", "循环", "流程", "运行时", "loop"),
     "content": "Agent 接收用户输入，由模型决定直接回答还是调用工具；执行工具后把结果交还模型，重复这一过程，直到得到最终回答或达到轮次上限。"},
    {"title": "工具调用", "keywords": ("工具", "调用", "函数", "tool", "schema"),
     "content": "工具注册时声明名称、用途和参数 Schema。模型选择工具并生成参数，运行时验证参数、执行工具，再把结果作为 tool 消息交还模型。"},
    {"title": "会话记忆", "keywords": ("会话", "记忆", "上下文", "压缩", "追问", "memory", "context"),
     "content": "每个 session 独立保存历史对话和待办。追问时加载当前 session 的近期消息；对话过长时将较早轮次压缩为摘要，可能丢失细节。"},
]


def search(args, _user, _session):
    query = args["query"].strip().lower()
    if not query:
        raise ValueError("查询词不能为空")
    hits = [dict(title=doc["title"], content=doc["content"])
            for doc in SEARCH_DOCS if any(word in query for word in doc["keywords"])]
    return {"source": "本地示例资料（非实时网页）", "hits": hits}


def validate_arguments(schema, args):
    if not isinstance(args, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    properties = schema["properties"]
    for name in schema.get("required", []):
        if name not in args:
            raise ValueError(f"缺少参数: {name}")
    for name, value in args.items():
        if name not in properties:
            raise ValueError(f"未知参数: {name}")
        field = properties[name]
        expected = field["type"]
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{name} 必须是字符串")
        if expected == "integer" and type(value) is not int:
            raise ValueError(f"{name} 必须是整数")
        if "enum" in field and value not in field["enum"]:
            raise ValueError(f"{name} 不在允许的选项中")


def built_in_tools(store):
    return [
        Tool("calculator", "计算算术表达式，支持括号和 + - * / **。",
             {"type": "object", "properties": {"expression": {"type": "string"}},
              "required": ["expression"], "additionalProperties": False}, calculate),
        Tool("search", "查询本地 Agent 示例资料，支持循环、工具调用和会话记忆；不提供实时网页信息。",
             {"type": "object", "properties": {"query": {"type": "string"}},
              "required": ["query"], "additionalProperties": False}, search),
        Tool("todo", "在当前会话中添加、查看或完成待办。id 是当前用户当前会话内从 1 开始的编号，完成后编号不变。list 每页最多 10 条；has_more 为真时用 next_after_id 作为 after_id 查询下一页。",
             {"type": "object", "properties": {
                 "action": {"type": "string", "enum": ["add", "list", "complete"]},
                 "text": {"type": "string"},
                 "id": {"type": "integer", "description": "仅用于 complete；使用当前会话 list 返回的待办编号。"},
                 "after_id": {"type": "integer", "description": "仅用于 list；非负整数，默认 0，从此 ID 之后继续查询。"}},
              "required": ["action"], "additionalProperties": False}, store.todo),
    ]
