#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把报告引用的本地图片上传到 picgo.net (Chevereto V4)，并把 md 引用改写为公网直链。

用法:
  1) 把 API key 写入同目录 .env:  PICG_KEY=xxxxxxxx
  2) python3 upload_picgo.py Qwen3.8-27B_四引擎性能对比报告.md

产出:
  - upload_map.json   本地文件 -> 直链 的映射
  - <原名>_发布版.md   引用全部替换为直链的副本（原 md 不改）
"""
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

BASE = Path(__file__).resolve().parent
API = "https://www.picgo.net/api/1/upload/"

def load_key() -> str:
    key = os.environ.get("PICG_KEY", "")
    if not key:
        envf = BASE / ".env"
        if envf.exists():
            for line in envf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("PICG_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("缺少 PICG_KEY：请在 .env 中写入 PICG_KEY=<你的key> 或设置环境变量")
    return key

IMG_RE = re.compile(r"!\[([^\]]*)\]\(<?([^)>]+?)>?(?:\s+\"[^\"]*\")?\)")

def upload(path: Path, key: str) -> str:
    """上传单个图片，返回 image.url 直链；失败抛 RuntimeError。"""
    import requests
    headers = {"X-Api-Key": key}
    url = f"{API}?key={urllib.parse.quote(key)}"
    with path.open("rb") as f:
        files = {"source": (path.name, f, "application/octet-stream")}
        resp = requests.post(url, headers=headers, files=files, timeout=120)
    body = resp.json()
    if resp.status_code != 200 or "image" not in body:
        raise RuntimeError(
            f"{path.name} 上传失败 HTTP {resp.status_code}: "
            f"{body.get('error') or body.get('status_txt')}")
    return body["image"]["url"]

def main() -> None:
    key = load_key()
    md = BASE / sys.argv[1] if len(sys.argv) > 1 else \
        BASE / "Qwen3.8-27B_四引擎性能对比报告_发布版.md"
    text = md.read_text(encoding="utf-8")

    imgs = {}
    for m in IMG_RE.finditer(text):
        ref = m.group(2).strip()
        if ref.lower().startswith(("http://", "https://", "data:")):
            continue
        p = (BASE / ref).resolve()
        if p not in imgs:
            imgs[p] = None

    mapping = {}
    if (BASE / "upload_map.json").exists():
        mapping = json.loads((BASE / "upload_map.json").read_text(encoding="utf-8"))

    todo = [p for p in imgs if str(p) not in mapping]
    print(f"共 {len(imgs)} 张图，待上传 {len(todo)} 张")
    for i, p in enumerate(todo, 1):
        try:
            url = upload(p, key)
            mapping[str(p)] = url
            print(f"[{i}/{len(todo)}] {p.name} -> {url}")
            (BASE / "upload_map.json").write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        except RuntimeError as e:
            print(f"[失败] {e}", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.3)

    missing = [p for p in imgs if str(p) not in mapping]
    if missing:
        print("仍有图片未上传成功:", *[m.name for m in missing], file=sys.stderr)
        sys.exit(1)

    def repl(m: re.Match) -> str:
        ref = m.group(2).strip()
        if ref.lower().startswith(("http://", "https://", "data:")):
            return m.group(0)
        return f"![{m.group(1)}]({mapping[str((BASE / ref).resolve())]})"

    new_text = IMG_RE.sub(repl, text)
    # 输入已是发布版时原地覆盖，否则另存发布版副本
    out = md if md.stem.endswith("_发布版") else md.with_name(md.stem + "_发布版.md")
    out.write_text(new_text, encoding="utf-8")
    print(f"已生成发布版: {out}")
    print(f"映射表: {BASE / 'upload_map.json'}")

if __name__ == "__main__":
    main()
