import gradio as gr
import requests
import os
API_BASE = "http://127.0.0.1:8000/api"

def text_qa(question):
    resp = requests.post(f"{API_BASE}/text_rag", data={"question": question})
    if resp.status_code == 200:
        return resp.json()["answer"]
    return "调用失败"

def multimodal_qa(image, question):
    with open(image, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/multimodal_rag",
            files={"image": f}, 
            data={"question": question})

    
    if resp.status_code == 200:
        data = resp.json()
        return data["answer"], data["image_description"]
    return "调用失败", ""
def batch_multimodal_qa(files, question):
    """处理多张图片，调用批量接口"""
    if not files:
        return "请至少上传一张图片", []
    
    files_list = []
    for file_path in files:
        
        filename = os.path.basename(file_path)
        files_list.append( ("images", (filename, open(file_path, "rb"), "image/jpeg") ))
        


    try:
        resp = requests.post(
            f"{API_BASE}/batch_multimodal_rag",
            files=files_list,
            data={"question": question}
            )
        if resp.status_code == 200:
            result = resp.json()#
          
            data_list = result.get("data", [])
            outputs = []
            for item in data_list:
                idx = item.get("img_index", 0) + 1
                name = item.get("filename", "未知")
                ans = item.get("answer", "无回答")
                outputs.append(f"### 图片 {idx}：{name}\n{ans}\n")
            return "\n".join(outputs)  
        
        else:
            return f"请求失败，状态码 {resp.status_code}", []
    except Exception as e:
        return f"请求异常: {str(e)}", []
    finally:
        for _, (_, f, _) in files_list:
            f.close()

with gr.Blocks(title="多模态智能问答系统") as demo:
   
    gr.Markdown("# 🏭 多模态智能问答系统")
    
    #==================第一个标签页==================
    with gr.Tab("文档问答"):#创建第1个标签页，名称为“文档问答”。
            q_in = gr.Textbox(label="输入问题")
            ans_out = gr.Markdown()
            btn2 = gr.Button("提问")
            btn2.click(text_qa, q_in, ans_out)
   
    #==================第二个标签页==================
    with gr.Tab("图文问答"):#创建一个标签页，名为“图文问答”。
       
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
       

    #==================第三个标签页==================
    #新增一个“批量图文问答”标签页
    with gr.Tab("批量图文问答"):
        with gr.Row():
            with gr.Column():
                file_input = gr.Files(label="上传零件图片（可多选）", file_types=[".jpg", ".jpeg", ".png"])
                q_input = gr.Textbox(label="统一问题", lines=2, placeholder="请输入要问所有图片的问题")
                btn = gr.Button("批量提问", variant="primary")
            with gr.Column():
                #
                ans_output = gr.Markdown(label="批量回答结果",value="等待提问...")
        btn.click(
            batch_multimodal_qa,
            inputs=[file_input, q_input],
            outputs=[ans_output]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

# python gradio_app.py