import gradio as gr
import requests
import time
import os
import json
API_BASE = "http://127.0.0.1:8000/api"

def text_qa(question):


    resp = requests.post(
        f"{API_BASE}/chat",
        files=[
            ("question", (None, question)),
            ("session_id", (None, f"gradio_{int(time.time() * 1000)}")),
        ]
    )


    if resp.status_code == 200:
        return resp.json()["answer"]
    return "调用失败"
def text_qa_stream(question):


    if not question or not question.strip():
        yield "请输入问题"
        return
    try:

        with requests.post(
            f"{API_BASE}/chat",
            files=[
                ("question", (None, question)),
                ("session_id", (None, f"gradio_{int(time.time() * 1000)}")),
                ("stream", (None, "true")),
            ],
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                yield f"请求失败，状态码 {resp.status_code}"
                return
            full = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "token":

                    full += evt.get("content", "")
                    yield full
                elif t == "tool_call":


                    full += f"\n\n> 🔧 调用工具：{evt.get('tool')}\n"
                    yield full
                elif t == "tool_result":

                    full += f"> 工具返回：{str(evt.get('result'))[:200]}\n\n"

                    yield full
                elif t == "done":
                    if evt.get("answer"):
                        yield evt["answer"]
                elif t == "error":
                    yield f"❌ 流式错误：{evt.get('message')}"
    except Exception as e:
        yield f"请求异常: {str(e)}"


def multimodal_qa(image, question):

    if not image:
        return "请上传图片", ""
    with open(image, "rb") as f:


        resp = requests.post(
            f"{API_BASE}/chat",
            files=[("images", (os.path.basename(image), f, "image/jpeg"))],
            data={"question": question, "session_id": f"gradio_{int(time.time() * 1000)}"})


    if resp.status_code == 200:
        result = resp.json()


        img_list = result.get("multimodalResults", [])
        if img_list and img_list[0].get("answer"):
            first = img_list[0]
            return first["answer"], first.get("image_description", "")

        return img_list[0].get("error", "处理失败") if img_list else "处理失败", ""
    return "调用失败", ""


def batch_multimodal_qa(files, question):

    if not files:
        return "请至少上传一张图片", []


    files_list = []
    for file_path in files:

        filename = os.path.basename(file_path)


        files_list.append( ("images", (filename, open(file_path, "rb"), "image/jpeg") ))


    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            files=files_list,
            data={"question": question, "session_id": f"gradio_{int(time.time() * 1000)}"},
            )
        if resp.status_code == 200:
            result = resp.json()

            data_list = result.get("multimodalResults", [])

            outputs = []
            for item in data_list:
                idx = item.get("img_index", 0) + 1
                name = item.get("filename", "未知")
                ans = item.get("answer", "无回答")
                outputs.append(f"### 图片 {idx}：{name}\n{ans}\n")
            return "\n".join(outputs)

        else:
            return f"请求失败，状态码 {resp.status_code}"
    except Exception as e:
        return f"请求异常: {str(e)}"
    finally:

        for _, (_, f, _) in files_list:
            f.close()

def multimodal_qa_stream(files, question):


    if not files:
        yield "请至少上传一张图片"
        return

    files_list = []
    for file_path in files:
        filename = os.path.basename(file_path)
        files_list.append(("images", (filename, open(file_path, "rb"), "image/jpeg")))
    try:

        with requests.post(
            f"{API_BASE}/chat",
            files=files_list,
            data={"question": question,
                  "session_id": f"gradio_{int(time.time() * 1000)}",
                  "stream": "true"},
            stream=True,
            timeout=300,
        ) as resp:


            if resp.status_code != 200:
                yield f"请求失败，状态码 {resp.status_code}"
                return


            parts = []
            for line in resp.iter_lines(decode_unicode=True):


                if not line or not line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "image_done":

                    idx = evt.get("img_index", len(parts))
                    fname = evt.get("filename") or "未知图片"
                    ans = evt.get("answer") or evt.get("error") or "（无回答）"
                    parts.append(f"### 图片 {idx + 1}：{fname}\n{ans}\n")
                    yield "".join(parts)
                elif evt.get("type") == "batch_summary":
                    if evt.get("batchSummary"):
                        parts.append(f"\n## 📊 批量缺陷统计\n{evt['batchSummary']}\n")
                        yield "".join(parts)
                elif evt.get("type") == "done":


                    if parts:
                        yield "".join(parts)
                    elif evt.get("answer"):
                        yield evt["answer"]
                elif evt.get("type") == "error":
                    yield f"❌ 流式错误：{evt.get('message')}"
    except Exception as e:
        yield f"请求异常: {str(e)}"
    finally:

        for _, (_, f, _) in files_list:


            f.close()


with gr.Blocks(title="多模态智能问答系统") as demo:

    gr.Markdown("# 🏭 多模态智能问答系统")


    with gr.Tab("文档问答"):
            q_in = gr.Textbox(label="输入问题")
            ans_out = gr.Markdown()
            btn2 = gr.Button("提问")
            btn2.click(text_qa, q_in, ans_out)


    with gr.Tab("图文问答"):

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="filepath", label="上传零件图片")


                q_input = gr.Textbox(label="问题", lines=2)
                btn = gr.Button("提问", variant="primary")

            with gr.Column():
                ans_output = gr.Markdown(label="回答")
                with gr.Accordion("图片解析", open=False):
                    desc_output = gr.Textbox(lines=5)
        btn.click(multimodal_qa, [img_input, q_input], [ans_output, desc_output])


    with gr.Tab("批量图文问答"):
        with gr.Row():
            with gr.Column():

                file_input = gr.Files(label="上传零件图片（可多选）", file_types=[".jpg", ".jpeg", ".png"])

                q_input = gr.Textbox(label="统一问题", lines=2, placeholder="请输入要问所有图片的问题")
                btn = gr.Button("批量提问", variant="primary")
            with gr.Column():

                ans_output = gr.Markdown(label="批量回答结果",value="等待提问...")
        btn.click(
            batch_multimodal_qa,
            inputs=[file_input, q_input],
            outputs=[ans_output]
            )


    with gr.Tab("流式图文问答"):
        with gr.Row():
            with gr.Column():
                file_input = gr.Files(label="上传零件图片（可多选，逐张流式返回）", file_types=[".jpg", ".jpeg", ".png"])
                q_input = gr.Textbox(label="统一问题", lines=2, placeholder="请输入要问所有图片的问题")
                btn = gr.Button("流式提问", variant="primary")
            with gr.Column():
                ans_output = gr.Markdown(label="流式回答结果", value="等待提问...")
        btn.click(
            multimodal_qa_stream,
            inputs=[file_input, q_input],
            outputs=[ans_output]
        )


    with gr.Tab("流式文档问答"):
        with gr.Row():
            with gr.Column():
                q_in = gr.Textbox(label="输入问题", lines=2, placeholder="请输入要问的问题，流式逐字返回")
                btn = gr.Button("流式提问", variant="primary")
            with gr.Column():
                ans_out = gr.Markdown(label="流式回答", value="等待提问...")
        btn.click(text_qa_stream, q_in, ans_out)
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
