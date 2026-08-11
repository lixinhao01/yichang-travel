#!/usr/bin/env python3
"""服务器端批量 OCR — 直接调 Ollama API，无额外依赖"""
import sys, re, time, json, base64, io, os
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import requests
from PIL import Image

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
TIMEOUT = 300

DST = Path("/data2/lxh/yichang_ocr")
PROGRESS = DST / "_ocr_progress.json"


def encode_image(path: str) -> str:
    """读取图片，webp→PNG，大图压缩"""
    im = Image.open(path).convert("RGB")
    maxdim = 1600
    if max(im.size) > maxdim:
        r = maxdim / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) > 3_500_000:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
    return base64.b64encode(data).decode()


def ocr_image(path: str) -> str:
    """调 Ollama 视觉模型 OCR"""
    b64 = encode_image(path)
    prompt = (
        "图片文字主要为中文。"
        "你是一个精准的OCR引擎。请完整、逐字提取这张图片中的【所有文字】，"
        "不要遗漏、不要翻译、不要总结。保留原始换行和列表结构，"
        "看不清的字用[?]标注。仅输出提取到的文字内容。"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "images": [b64], "stream": False,
              "options": {"temperature": 0.1}},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    out = data.get("response", "")
    if not out and data.get("error"):
        raise RuntimeError(data["error"])
    return out.strip()


# 等待模型就绪
print(f"检查模型 {MODEL}...")
for attempt in range(60):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any(MODEL in m for m in models):
            print(f"模型就绪: {MODEL}")
            break
    except:
        pass
    print(f"  等待模型下载中... ({attempt+1}/60)")
    time.sleep(30)
else:
    print("模型未就绪，退出")
    sys.exit(1)

# 预热
print("预热模型...")
try:
    # 用最小图片预热
    test_img = DST / "img" / sorted((DST / "img").iterdir())[0].name
    test_files = list(test_img.glob("*.webp")) + list(test_img.glob("*.jpg")) + list(test_img.glob("*.png"))
    if test_files:
        ocr_image(str(test_files[0]))
        print("预热完成")
except Exception as e:
    print(f"预热失败: {e}")

# 加载进度
if PROGRESS.exists():
    done = json.loads(PROGRESS.read_text(encoding="utf-8"))
else:
    done = {}

# 只处理数字编号的 .md 文件
md_files = sorted([f for f in DST.glob("*.md") if f.name[0].isdigit() and not f.name.startswith("00_")])
print(f"\n共 {len(md_files)} 个文件待处理")

total_ocr = 0
total_skip = 0
errors = []
t_start = time.time()

for fi, f in enumerate(md_files, 1):
    text = f.read_text(encoding="utf-8")

    img_pattern = re.compile(r'(!\[图(\d+)\]\((img/[^)]+)\))')
    matches = list(img_pattern.finditer(text))
    if not matches:
        continue

    file_key = f.name
    if file_key not in done:
        done[file_key] = {}

    modified = False
    for m in reversed(matches):
        img_num = m.group(2)
        img_rel = m.group(3)
        img_path = str(DST / img_rel)

        if img_rel in done[file_key] and done[file_key][img_rel]:
            total_skip += 1
            continue

        if not Path(img_path).exists():
            errors.append(f"{f.name}: {img_rel} 不存在")
            continue

        try:
            t0 = time.time()
            result = ocr_image(img_path)
            dt = time.time() - t0

            if not result or len(result) < 3:
                done[file_key][img_rel] = ""
                total_skip += 1
                continue

            # 插入 OCR 文本到图片行后
            insert_text = f"\n> [OCR] {result}"
            line_end = m.end()
            rest = text[line_end:]
            next_nl = rest.find("\n")
            insert_pos = line_end + next_nl if next_nl >= 0 else len(text)
            text = text[:insert_pos] + insert_text + text[insert_pos:]

            done[file_key][img_rel] = result[:100]
            total_ocr += 1
            modified = True
            print(f"  [{fi}/{len(md_files)}] {f.name[:30]} 图{img_num} ({dt:.1f}s): {result[:40]}...")

        except Exception as e:
            errors.append(f"{f.name} 图{img_num}: {e}")
            print(f"  [{fi}/{len(md_files)}] {f.name[:30]} 图{img_num} ❌ {e}")

    if modified:
        f.write_text(text, encoding="utf-8")

    if fi % 5 == 0:
        PROGRESS.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

PROGRESS.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

elapsed = time.time() - t_start
print(f"\n=== 完成 ({elapsed/60:.0f}分钟) ===")
print(f"OCR: {total_ocr} 张, 跳过: {total_skip} 张, 错误: {len(errors)} 个")
if errors:
    print("错误（前20条）:")
    for e in errors[:20]:
        print(f"  {e}")
