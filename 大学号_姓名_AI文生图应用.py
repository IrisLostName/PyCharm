import os
import json
import gradio as gr
from openai import OpenAI

# 1. 配置阿里云百炼客户端
# 请替换为你的实际 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

HISTORY_FILE = "chat_history.json"

def load_history():
    """从本地文件加载历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(messages):
    """保存对话到本地"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

# 2. 对话逻辑
def predict(user_input, chatbot, system_prompt):
    if not user_input:
        yield "", chatbot
        return

    # 构建发送给 API 的消息序列
    # chatbot 格式为: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in chatbot:
        api_messages.append(msg)

    # 添加当前用户输入
    current_user_msg = {"role": "user", "content": user_input}
    api_messages.append(current_user_msg)

    # 更新界面：先显示用户的问题，并为 AI 回复留出占位
    chatbot.append(current_user_msg)
    chatbot.append({"role": "assistant", "content": ""})

    try:
        # 使用 OpenAI SDK 进行流式调用
        completion = client.chat.completions.create(
            model="qwen-plus",  # 或者 qwen-turbo
            messages=api_messages,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_response = ""
        for chunk in completion:
            # 过滤掉 usage 等非内容数据块
            if len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    chatbot[-1]["content"] = full_response
                    yield "", chatbot  # 实时刷新界面

        # 对话结束后保存
        save_history(chatbot)

    except Exception as e:
        chatbot[-1]["content"] = f"❌ 发生错误: {str(e)}"
        yield "", chatbot

def clear_chat():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return [], ""

# 3. 界面布局
with gr.Blocks(title="通义千问助手") as demo:
    gr.Markdown("""
    # 🤖 通义千问智能助手 (SDK版)
    **功能：** 多轮对话、自定义人设、本地记录保存、流式响应。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            system_input = gr.Textbox(
                label="机器人人设设定",
                value="你是一个通晓古今、说话幽默的 AI 助手。",
                lines=5
            )
            clear_btn = gr.Button("🗑️ 清空所有历史", variant="stop")
            gr.Markdown("---")
            gr.Markdown("**提示：** 历史记录将自动保存在同一目录下的 `chat_history.json` 中。")

        with gr.Column(scale=4):
            # 显式使用 type="messages" 以匹配 OpenAI 格式
            chatbot = gr.Chatbot(
                label="对话窗口",
                value=load_history(),
                height=550
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入框",
                    placeholder="请输入您想问的问题，按回车提交...",
                    show_label=False,
                    scale=8
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

    # 4. 事件绑定
    # 提交逻辑
    msg_input.submit(predict, [msg_input, chatbot, system_input], [msg_input, chatbot])
    submit_btn.click(predict, [msg_input, chatbot, system_input], [msg_input, chatbot])
    # 清空逻辑
    clear_btn.click(clear_chat, None, [chatbot, msg_input])

if __name__ == "__main__":
    # 启动
    demo.queue().launch()
