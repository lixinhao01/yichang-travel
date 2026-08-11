#!/usr/bin/env python3
"""批量 OCR 小红书图片，将提取文字插入对应 .md 文件图片下方"""
import sys, re, time, json
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

# 导入 local_vision_server 的函数
sys.path.insert(0, str(Path.home() / ".claude" / "mcp_servers" / "local-vision"))
from local_vision_server import extract_text, _encode_image

DST = Path(r"C:\Users\Administrator\Desktop\博士阶段（总）\笔记\笔记\宜昌旅游\小红书原始数据\正文")

# 只处理数字编号的 .md 文件
md_files = sorted([f for f in DST.glob("*.md") if f.name[0].isdigit() and not f.name.startswith("00_")])
print(f"共 {len(md_files)} 个文件")

# 进度文件（断点续传）
PROGRESS = DST / "_ocr_progress.json"
if PROGRESS.exists():
    done = json.loads(PROGRESS.read_text(encoding="utf-8"))
else:
    done = {}

total_ocr = 0
total_skip = 0
errors = []

for fi, f in enumerate(md_files, 1):
    text = f.read_text(encoding="utf-8")

    # 找所有图片行: ![图N](img/xxx/NN.webp)
    img_pattern = re.compile(r'(!\[图(\d+)\]\((img/[^)]+)\))')
    matches = list(img_pattern.finditer(text))

    if not matches:
        continue

    file_key = f.name
    if file_key not in done:
        done[file_key] = {}

    # 从后往前替换（避免偏移）
    modified = False
    for m in reversed(matches):
        full_line = m.group(1)
        img_num = m.group(2)
        img_rel = m.group(3)
        img_path = str(DST / img_rel)

        # 跳过已 OCR 的
        cache_key = img_rel
        if cache_key in done[file_key] and done[file_key][cache_key]:
            total_skip += 1
            continue

        # 检查图片文件存在
        if not Path(img_path).exists():
            errors.append(f"{f.name}: 图片不存在 {img_rel}")
            continue

        # OCR
        try:
            t0 = time.time()
            ocr_text = extract_text(img_path, language="zh")
            dt = time.time() - t0

            # 清理 OCR 结果
            ocr_text = ocr_text.strip()
            if not ocr_text or len(ocr_text) < 3:
                done[file_key][cache_key] = ""
                total_skip += 1
                continue

            # 判断是否只是纯图片（无文字内容）
            skip_words = ["图片", "照片", "拍摄", "风景", "画面", "场景", "没有文字", "无法识别"]
            if len(ocr_text) < 20 and any(w in ocr_text for w in skip_words):
                done[file_key][cache_key] = ""
                total_skip += 1
                continue

            # 在图片行后插入 OCR 文本
            insert_text = f"\n> [OCR] {ocr_text}"
            # 找到图片行在 text 中的位置，在其后插入
            line_start = m.start()
            line_end = m.end()
            # 找到该行的末尾（包括换行符）
            rest = text[line_end:]
            next_newline = rest.find("\n")
            if next_newline >= 0:
                insert_pos = line_end + next_newline
            else:
                insert_pos = len(text)

            text = text[:insert_pos] + insert_text + text[insert_pos:]
            done[file_key][cache_key] = ocr_text[:100]  # 缓存摘要
            total_ocr += 1
            modified = True
            print(f"  [{fi}/{len(md_files)}] 图{img_num} ({dt:.1f}s): {ocr_text[:50]}...")

        except Exception as e:
            errors.append(f"{f.name} 图{img_num}: {e}")
            print(f"  [{fi}/{len(md_files)}] 图{img_num} ❌ {e}")

    # 写回文件
    if modified:
        f.write_text(text, encoding="utf-8")

    # 每 10 个文件保存一次进度
    if fi % 10 == 0:
        PROGRESS.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

# 最终保存
PROGRESS.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n=== 完成 ===")
print(f"OCR: {total_ocr} 张, 跳过: {total_skip} 张, 错误: {len(errors)} 个")
if errors:
    print("错误列表:")
    for e in errors[:20]:
        print(f"  {e}")
