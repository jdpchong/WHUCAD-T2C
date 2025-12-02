# LLM/test_qwen2_vl.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 1. 加载 processor（文本 + 图像）
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 2. 加载多模态模型
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # 3. 准备一张测试图片（放一张 test.png 在 LLM/ 下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "test.png")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Please put a test.png under {script_dir}")

    image = Image.open(image_path).convert("RGB")

    # 4. 写一个用户问题
    prompt = "Please describe the main objects and geometric structures shown in this image."

    # 5. 用 Qwen2-VL 的 chat_template 构造带 image 占位符的文本
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # 这里是占位符，真正的图像在 images 参数里传
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 6. 用 processor 打包文本 + 图像
    inputs = processor(
        text=[chat_text],        # batch size 1
        images=[image],          # 与上面 "type": "image" 数量一致
        return_tensors="pt",
    ).to(DEVICE)

    # 7. 生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

    # 8. 解码
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 只保留 assistant 后面的内容
    marker = "assistant\n"
    if marker in generated_text:
        answer = generated_text.split(marker, 1)[1].strip()
    else:
        answer = generated_text.strip()

    print(answer)


if __name__ == "__main__":
    main()
