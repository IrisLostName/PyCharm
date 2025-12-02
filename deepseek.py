from openai import OpenAI

client = OpenAI(
    # 如果没有配置环境变量，请⽤阿⾥云百炼 API Key 替换：api_key="sk-xxx"
    ####sk-9b7fdff3e2674d34ae8bf409d39205e7
    api_key="sk-9b7fdff3e2674d34ae8bf409d39205e7",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
topic="家国情怀"
prompt="请生成包含三本****"

messages = [{"role": "user", "content": prompt}]

# 此处以 deepseek-v3.2-exp 为例，可按需更换模型名称为 deepseek-v3.1、deepseek-v3 或 deepseek-r1
completion = client.chat.completions.create(model="deepseek-v3.2-exp",messages=messages,extra_body={"enable_thinking": True}, stream=True, stream_options={"include_usage": True },)

reasoning_content = ""  # 完整思考过程
answer_content = ""  # 完整回复
is_answering = False  # 是否进⼊回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
        print(chunk.usage)
        continue
    delta = chunk.choices[0].delta

# 只收集思考内容
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content
    # 收到 content，开始进⾏回复
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content
