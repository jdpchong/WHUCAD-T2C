import os
import json
from typing import Optional
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===== 配置 =====

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# 默认用 GPU，没有就退回 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

JSON_ROOT = os.path.join(PROJECT_ROOT, "data", "json")
MULTIVIEW_ROOT = os.path.join(PROJECT_ROOT, "data", "multiview")
TEXT_ROOT = os.path.join(PROJECT_ROOT, "data", "text")

MAX_NEW_TOKENS = 192
TEMPERATURE = 0.3
TOP_P = 0.9

# 如需只调试前 N 个样本，改成整数；None 表示全部
MAX_SAMPLES: Optional[int] = 10 #None

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

def load_model():
    print(f"[INFO] Loading model: {MODEL_NAME} on {DEVICE}")
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
    return processor, model


def build_prompt(cad_json_dict):
    cad_json_str = json.dumps(cad_json_dict, indent=2)
    return PROMPT_TEMPLATE.format(cad_json=cad_json_str)


def pick_combined_image_for_json(json_path: str) -> Optional[Image.Image]:
    """
    json: data/json/0000/00000000.json
    img:  data/multiview/0000/00000000/combined*.png

    只取一张以 'combined' 开头的 png；
    找不到则返回 None。
    """
    rel_path = os.path.relpath(json_path, JSON_ROOT)   # "0000/00000000.json"
    rel_no_ext, _ = os.path.splitext(rel_path)         # "0000/00000000"
    img_dir = os.path.join(MULTIVIEW_ROOT, rel_no_ext)

    if not os.path.isdir(img_dir):
        print(f"[WARN] multiview directory not found for json: {json_path}")
        return None

    candidates = [
        f for f in sorted(os.listdir(img_dir))
        if f.lower().endswith(".png") and f.lower().startswith("combined")
    ]
    if not candidates:
        print(f"[WARN] no combined*.png found for json: {json_path}")
        return None

    img_path = os.path.join(img_dir, candidates[0])
    try:
        img = Image.open(img_path).convert("RGB")
        # 可选：缩小一点，减轻显存压力
        img = img.resize((512, 512))
        return img
    except Exception as e:
        print(f"[WARN] failed to open image {img_path}: {e}")
        return None


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


def generate_description(processor, model, prompt: str, image: Image.Image) -> str:
    """
    使用 Qwen2-VL 生成一段描述，只用一张 combined 图。
    """
    contents = [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": contents}]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return postprocess_answer(raw)


# ===== 主流程 =====

def main():
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("JSON_ROOT    :", JSON_ROOT)
    print("MULTIVIEW_ROOT:", MULTIVIEW_ROOT)
    print("TEXT_ROOT    :", TEXT_ROOT)
    print("DEVICE       :", DEVICE)

    os.makedirs(TEXT_ROOT, exist_ok=True)

    processor, model = load_model()

    # 收集所有 json
    json_files = []
    for dirpath, _, filenames in os.walk(JSON_ROOT):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                json_files.append(os.path.join(dirpath, fname))

    json_files = sorted(json_files)
    print(f"[INFO] Found {len(json_files)} JSON files.")

    if MAX_SAMPLES is not None:
        json_files = json_files[:MAX_SAMPLES]
        print(f"[INFO] Limiting to first {len(json_files)} samples.")

    for json_path in tqdm(json_files):
        rel_path = os.path.relpath(json_path, JSON_ROOT)  # e.g. "0000/00000000.json"
        rel_no_ext, _ = os.path.splitext(rel_path)
        out_path = os.path.join(TEXT_ROOT, rel_no_ext + ".txt")

        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # 已经有文本了就跳过
        if os.path.exists(out_path):
            continue

        # 读 JSON，失败就跳过
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cad_json = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load json {json_path}: {e}")
            continue

        # 找对应 combined 图，找不到就跳过
        image = pick_combined_image_for_json(json_path)
        if image is None:
            # 这里就是你说的“找到 json 找不到 png 就直接跳过”
            continue

        prompt = build_prompt(cad_json)

        # 单个样本错误不影响后续
        try:
            desc = generate_description(processor, model, prompt, image)
        except Exception as e:
            print(f"[WARN] failed to generate for {json_path}: {e}")
            continue

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(desc)
        except Exception as e:
            print(f"[WARN] failed to write {out_path}: {e}")
            continue

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
