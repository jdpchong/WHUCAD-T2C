import os
import json

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===== 配置 =====

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# GPU 版本：优先用 cuda，万一没有就退回 cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

JSON_ROOT = os.path.join(PROJECT_ROOT, "data", "json")
MULTIVIEW_ROOT = os.path.join(PROJECT_ROOT, "data", "multiview")

MAX_NEW_TOKENS = 192
TEMPERATURE = 0.3
TOP_P = 0.9

PROMPT_TEMPLATE = (
    "You are an expert CAD engineer.\n\n"
    "You will be given a CAD modeling command sequence in JSON format and a rendered "
    "view of the final part from multiple combined angles.\n\n"
    "Write ONE natural English paragraph (5–8 sentences) that describes the external "
    "shape and main geometric features of the part as a human would describe it in "
    "everyday use (for example: \"a rectangular table with six legs\").\n\n"
    "Strict rules:\n"
    "- Do NOT mention images, views, pictures, photos, renders, screenshots, cameras, or JSON.\n"
    "- Do NOT use phrases like \"this image shows\", \"the picture shows\", \"in the image\".\n"
    "- Do NOT mention colors.\n"
    "- Do NOT use bullet points or numbered lists.\n"
    "- Output exactly ONE paragraph.\n\n"
    "Here is the CAD command sequence (in JSON format):\n"
    "{cad_json}\n\n"
    "Now write one concise but informative paragraph following the rules above.\n"
)


# ===== 工具函数 =====

def find_first_json(json_root: str) -> str:
    """在 data/json 下找到第一个 .json 文件路径"""
    for dirpath, _, filenames in os.walk(json_root):
        for fname in sorted(filenames):
            if fname.lower().endswith(".json"):
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"No .json file found under {json_root}")


def pick_combined_image_for_json(json_path: str) -> str:
    """
    json: data/json/0000/00000000.json
    imgs: data/multiview/0000/00000000/combined*.png

    只选一张以 'combined' 开头的 png（比如 combined.png / combined_white.png）。
    """
    rel_path = os.path.relpath(json_path, JSON_ROOT)   # "0000/00000000.json"
    rel_no_ext, _ = os.path.splitext(rel_path)         # "0000/00000000"
    img_dir = os.path.join(MULTIVIEW_ROOT, rel_no_ext)

    if not os.path.isdir(img_dir):
        print(f"[WARN] multiview directory not found: {img_dir}")
        return None

    candidates = [
        f for f in sorted(os.listdir(img_dir))
        if f.lower().endswith(".png") and f.lower().startswith("combined")
    ]
    if not candidates:
        print(f"[WARN] no combined*.png found in {img_dir}")
        return None

    return os.path.join(img_dir, candidates[0])


def postprocess_answer(raw_text: str) -> str:
    """
    1. 去掉前面的 system/user 等前缀，只保留 assistant 部分。
    2. 把换行压成一段。
    3. 去掉常见“图像说明”前缀。
    """
    marker = "assistant\n"
    if marker in raw_text:
        text = raw_text.split(marker, 1)[1].strip()
    else:
        text = raw_text.strip()

    text = " ".join(text.split())

    bad_phrases = [
        "This image shows", "This picture shows", "This figure shows",
        "The image shows", "The picture shows", "The figure shows",
        "In the image,", "In this image,", "In the picture,", "In this picture,"
    ]
    for bp in bad_phrases:
        text = text.replace(bp, "").strip()

    return text


# ===== 主流程 =====

def main():
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("JSON_ROOT    :", JSON_ROOT)
    print("MULTIVIEW_ROOT:", MULTIVIEW_ROOT)
    print("DEVICE       :", DEVICE)

    # 1. 选一个 JSON 样本
    json_path = find_first_json(JSON_ROOT)
    print("\n[INFO] Using JSON file:", json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        cad_json = json.load(f)

    prompt = PROMPT_TEMPLATE.format(cad_json=json.dumps(cad_json, indent=2))

    # 2. 选 combined*.png
    img_path = pick_combined_image_for_json(json_path)
    if img_path is None:
        print("[WARN] No combined*.png found, will run with JSON only.")
        image = None
    else:
        print("[INFO] Using combined image:", img_path)
        image = Image.open(img_path).convert("RGB")
        # 可选：稍微缩小，减轻显存压力
        image = image.resize((512, 512))

    # 3. 加载模型（优先用 GPU）
    print("\n[INFO] Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if DEVICE == "cuda":
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        model.to(DEVICE)

    model.eval()

    # 4. 构造 messages + chat_text
    contents = []
    if image is not None:
        contents.append({"type": "image"})
    contents.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": contents}]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 5. 打包输入
    if image is None:
        inputs = processor(text=[chat_text], return_tensors="pt")
    else:
        inputs = processor(text=[chat_text], images=[image], return_tensors="pt")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 6. 生成
    print("\n[INFO] Generating description...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    desc = postprocess_answer(raw)

    print("\n======== RAW OUTPUT (first 400 chars) ========")
    print(raw[:400])
    print("\n======== FINAL DESCRIPTION ========")
    print(desc)
    print("===================================")


if __name__ == "__main__":
    main()
