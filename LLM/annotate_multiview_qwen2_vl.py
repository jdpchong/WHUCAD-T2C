import os
import json
from typing import List
from tqdm import tqdm
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Multi-modal model name (Qwen2-VL)
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
# Paths: infer project root based on this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

JSON_ROOT = os.path.join(PROJECT_ROOT, "data", "json")
MULTIVIEW_ROOT = os.path.join(PROJECT_ROOT, "data", "multiview")
TEXT_ROOT = os.path.join(PROJECT_ROOT, "data", "text")

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
TOP_P = 0.9

MAX_IMAGE_SIZE = 640      # 最长边 640 像素


PROMPT_TEMPLATE = (
    "You are an expert CAD engineer.\n\n"
    "You will be given:\n"
    "1) A CAD modeling command sequence in JSON format. Each step has a type_name such as\n"
    "   Line, Arc, Circle, Ext, Rev, Pocket, Shell, Chamfer, Fillet, Draft, Mirror, Hole, Topo, Select, etc.,\n"
    "   and its used parameters.\n"
    "2) A single multiview rendering of the final part that shows the geometry from several angles.\n\n"
    "Your task is to write ONE natural English sentence-style paragraph that describes\n"
    "the external shape and main geometric features of the part as a human would describe it\n"
    "in everyday use (for example: \"a rectangular table with six legs\" or\n"
    "\"a short cylinder with a large central hole and four small holes around it\").\n\n"
    "Strict requirements:\n"
    "- Do NOT mention images, views, pictures, photos, renders, screenshots, cameras, or JSON.\n"
    "- Do NOT use phrases like \"this image shows\", \"the picture shows\", \"in the image\", \"in the figure\".\n"
    "- Do NOT mention colors at all.\n"
    "- Do NOT use bullet points or numbered lists.\n"
    "- Output exactly ONE paragraph.\n"
    "- Focus on overall shape, main bodies, holes, pockets, slots, bosses, ribs,\n"
    "  chamfers, fillets and other visible geometric features.\n\n"
    "Here is the CAD command sequence (in JSON format):\n"
    "{cad_json}\n\n"
    "Now write one concise but informative paragraph following the rules above.\n"
)


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def build_prompt(cad_json_dict):
    cad_json_str = json.dumps(cad_json_dict, indent=2)
    return PROMPT_TEMPLATE.format(cad_json=cad_json_str)


def _resize_image(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """按最长边缩放到不超过 max_size，等比缩放。"""
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    return img


def find_multiview_image_for_json(json_path: str) -> List[Image.Image]:
    """
    data/json/0000/00000000.json
      -> data/multiview/0000/00000000/combined*.png

    规则：
      - 优先找文件名以 'combined' 开头的 .png（例如 combined_white.png）
      - 如果不存在 combined*.png，则退而求其次：用目录里的第一张 .png
      - 返回值仍然是 List[Image]，但只含 1 张 multiview 图
    """
    rel_path = os.path.relpath(json_path, JSON_ROOT)  # "0000/00000000.json"
    rel_no_ext, _ = os.path.splitext(rel_path)        # "0000/00000000"

    img_dir = os.path.join(MULTIVIEW_ROOT, rel_no_ext)
    images: List[Image.Image] = []

    if not os.path.isdir(img_dir):
        return images

    files = sorted(os.listdir(img_dir))
    combined_pngs = [
        f for f in files
        if f.lower().endswith(".png") and f.lower().startswith("combined")
    ]

    target_file = None
    if combined_pngs:
        target_file = combined_pngs[0]
    else:
        # 没有 combined 就随便找一张 png 作兜底
        pngs = [f for f in files if f.lower().endswith(".png")]
        if pngs:
            target_file = pngs[0]

    if target_file is None:
        return images

    img_path = os.path.join(img_dir, target_file)
    img = Image.open(img_path).convert("RGB")
    img = _resize_image(img, MAX_IMAGE_SIZE)
    images.append(img)

    return images


def postprocess_answer(raw_text: str) -> str:
    """
    1. 去掉前面的 system/user 等前缀，只保留 assistant 部分。
    2. 把换行压成一个段落。
    3. 简单清理常见的“图像说明”前缀。
    """
    marker = "assistant\n"
    if marker in raw_text:
        text = raw_text.split(marker, 1)[1].strip()
    else:
        text = raw_text.strip()

    # 把多行变成一段
    text = " ".join(text.split())

    bad_phrases = [
        "This image shows", "This picture shows", "This figure shows",
        "The image shows", "The picture shows", "The figure shows",
        "In the image,", "In this image,", "In the picture,", "In this picture,"
    ]
    for bp in bad_phrases:
        text = text.replace(bp, "").strip()

    return text


def generate_description(processor, model, prompt: str, images: List[Image.Image]) -> str:
    """
    Qwen2-VL 标准流程：
      1) messages 里放 {"type": "image"} 占位 + {"type": "text"} 提示；
      2) apply_chat_template 得到 chat_text；
      3) processor(text=[chat_text], images=[...]) 打包；
      4) model.generate → decode → 保留一段话。
    """
    contents = []
    for _ in images:
        contents.append({"type": "image"})
    contents.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": contents,
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if len(images) == 0:
        inputs = processor(
            text=[chat_text],
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[chat_text],
            images=images,  # 这里只有 1 张 multiview 图
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate WHUCAD samples with Qwen2-VL using multiview images."
    )
    parser.add_argument(
        "--subdirs",
        type=str,
        default="",
        help=(
            "Comma-separated list of subdirectory names under data/json to process, "
            "e.g. '0000,0003,0007'. If empty, process all."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit on number of samples to process (0 means no limit).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("JSON_ROOT   :", JSON_ROOT)
    print("MULTIVIEW   :", MULTIVIEW_ROOT)
    print("TEXT_ROOT   :", TEXT_ROOT)
    print("DEVICE      :", DEVICE)

    os.makedirs(TEXT_ROOT, exist_ok=True)

    # 解析 subdirs 参数
    subdirs = []
    if args.subdirs.strip():
        subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
        print("Only processing subdirs under data/json:", subdirs)
    else:
        print("Processing ALL subdirs under data/json.")

    processor, model = load_model()

    # 收集所有 json 文件（或指定子目录中的 json）
    json_files = []

    if subdirs:
        for sd in subdirs:
            root = os.path.join(JSON_ROOT, sd)
            if not os.path.isdir(root):
                print(f"[WARN] Subdir does not exist, skip: {root}")
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if fname.lower().endswith(".json"):
                        json_files.append(os.path.join(dirpath, fname))
    else:
        for dirpath, _, filenames in os.walk(JSON_ROOT):
            for fname in filenames:
                if fname.lower().endswith(".json"):
                    json_files.append(os.path.join(dirpath, fname))

    json_files = sorted(json_files)
    print(f"Found {len(json_files)} JSON files (before max-samples limit).")

    if args.max_samples > 0:
        json_files = json_files[: args.max_samples]
        print(f"Limiting to first {len(json_files)} samples due to --max-samples.")

    # 主循环：支持断点恢复（已有 txt 直接跳过）
    for json_path in tqdm(json_files):
        rel_path = os.path.relpath(json_path, JSON_ROOT)  # e.g. "0000/00000000.json"
        rel_no_ext, _ = os.path.splitext(rel_path)
        out_path = os.path.join(TEXT_ROOT, rel_no_ext + ".txt")

        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # 断点恢复：如果已经有 txt，直接跳过
        if os.path.exists(out_path):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cad_json = json.load(f)

            views = find_multiview_image_for_json(json_path)
            prompt = build_prompt(cad_json)
            desc = generate_description(processor, model, prompt, views)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(desc)

        except Exception as e:
            print(f"\n[ERROR] Failed on: {json_path}")
            print(repr(e))

        # 每个样本结束后手动释放显存
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("All done.")


if __name__ == "__main__":
    main()
