
import base64
from io import BytesIO
from PIL import Image
import requests
from core.config import API_KEY, BASE_URL, VL_MODEL


import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

def image_to_base64(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        buf = BytesIO()
       
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    


logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),  
    wait=wait_exponential(multiplier=1, min=1, max=10), 
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,  
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def parse_image(img_path, prompt):
    """通用视觉大模型调用"""
    img_b64 = image_to_base64(img_path)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": VL_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        "temperature": 0.1
    }
    resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]