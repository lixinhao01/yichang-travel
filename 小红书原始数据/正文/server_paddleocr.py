#!/usr/bin/env python3
"""服务器端 PaddleOCR 批量提取图片文字，插入 .md 文件"""
import sys, re, time, json, os
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

DST = Path("/data2/lxh/yichang_ocr")
PROGRESS = DST / "_ocr_progress.json"

# 初始化 PaddleOCR
print("加载 PaddleOCR...")
t0 = time.time()
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)
print(f"模型加载: {time.time()-t0:.1f}s")

def ocr_image(path: str) -> str:
    """调 PaddleOCR 提取文字"""
    result = ocr.ocr(path, cls=True)
    if not result or not result[0]:
        return ""
    lines = []
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        if conf > 0.5:  # 置信度 > 50% 才保留
            lines.append(text)
    return "\n".join(lines)


# 加载进度
if PROGRESS.exists():
    done = json.loads(PROGRESS.read_text(encoding="utf-8"))
else:
    done = {}

md_files = sorted([f for f in DST.glob("*.md") if f.name[0].isdigit() and not f.name.startswith("00_")])
print(f"共 {len(md_files)} 个文件待处理")

# 预热
print("预热...")
test_dir = DST / "img" / sorted((DST / "img").iterdir())[0].name
test_files = list(test_dir.glob("*.webp")) + list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
if test_files:
    ocr_image(str(test_files[0]))
    print("预热完成")

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

            if not result or len(result) < 2:
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
            print(f"  [{fi}/{len(md_files)}] {f.name[:30]} 图{img_num} ({dt:.1f}s): {result[:50]}...")

        except Exception as e:
            errors.append(f"{f.name} 图{img_num}: {e}")
            print(f"  [{fi}/{len(md_files)}] {f.name[:30]} 图{img_num} ❌ {e}")

    if modified:
        f.write_text(text, encoding="utf-8")

    if fi % 10 == 0:
        PROGRESS.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

PROGRESS.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

elapsed = time.time() - t_start
print(f"\n=== 完成 ({elapsed/60:.1f}分钟) ===")
print(f"OCR: {total_ocr} 张, 跳过: {total_skip} 张, 错误: {len(errors)} 个")
if errors:
    print("错误（前20条）:")
    for e in errors[:20]:
        print(f"  {e}")
