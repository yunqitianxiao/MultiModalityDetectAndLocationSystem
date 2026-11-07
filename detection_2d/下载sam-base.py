#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 SAM (Segment Anything Model) ViT-Base 权重
  sam_vit_b_01ec64.pth  (~358 MB, 官方 dl.fbaipublicfiles.com)

用法:
    python3 下载sam-base.py                      # 默认下载 ViT-B 到 ./sam_models/
    python3 下载sam-base.py --url <URL>          # 自定义下载地址
    python3 下载sam-base.py --verify <SHA256>    # 下载后校验哈希(不匹配则报错)
    python3 下载sam-base.py --check              # 只打印已存在文件的 SHA256

特性:
    - 自动读取环境 http/https 代理(HTTP_PROXY/HTTPS_PROXY),
      避免被 ALL_PROXY=socks 代理坑到
    - 支持断点续传(中断后重跑从断点继续)
    - 下载完成后打印 SHA256 便于人工校验
"""
import argparse
import hashlib
import os
import sys
import urllib.request

# 官方权重地址 (ViT-Base)
DEFAULT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
FILENAME = "sam_vit_b_01ec64.pth"          # ViT-B 权重文件名
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_models")
CHUNK = 256 * 1024


def build_opener():
    """构建 urllib opener: 只透传 http/https 代理, 忽略 socks(ALL_PROXY)"""
    proxy = {}
    for scheme in ("http", "https"):
        v = os.environ.get(scheme.upper() + "_PROXY") or os.environ.get(scheme + "_proxy")
        if v:
            proxy[scheme] = v
    if proxy:
        print(f"[下载] 使用代理: {proxy}")
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxy))


def download(url, dest):
    """下载(带断点续传 + 进度条)"""
    tmp = dest + ".part"
    opener = build_opener()

    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    with opener.open(req) as resp:
        total = existing + int(resp.headers.get("Content-Length", 0))
        mode = "ab" if existing else "wb"
        downloaded = existing
        last_reported = 0
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_reported > 1_000_000:   # 每 ~1MB 刷一次进度
                    last_reported = downloaded
                    pct = downloaded / total * 100 if total else 0
                    print(f"\r  已下载 {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.1f}%)",
                          end="", flush=True)
    os.replace(tmp, dest)
    print()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="下载 SAM ViT-Base 权重")
    ap.add_argument("--url", default=DEFAULT_URL, help="下载地址(默认官方 URL)")
    ap.add_argument("--verify", default=None, help="期望的 SHA256, 下载后校验")
    ap.add_argument("--check", action="store_true", help="只打印已下载文件的 SHA256, 不下载")
    args = ap.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    dest = os.path.join(DOWNLOAD_DIR, FILENAME)

    if args.check:
        if os.path.exists(dest):
            print(f"[校验] {dest}")
            print(f"  SHA256: {sha256(dest)}")
        else:
            print(f"[校验] 文件不存在: {dest}")
        return

    # 已完整下载则跳过
    if os.path.exists(dest) and not os.path.exists(dest + ".part"):
        print(f"[下载] 已存在, 跳过: {dest}")
        if args.verify:
            h = sha256(dest)
            ok = (h == args.verify.lower())
            print(f"  SHA256: {h}")
            print(f"  校验: {'✓ 通过' if ok else '✗ 不匹配!'}")
            if not ok:
                sys.exit(1)
        return

    print(f"[下载] {FILENAME}")
    print(f"  目标: {dest}")
    print(f"  来源: {args.url}")
    download(args.url, dest)

    h = sha256(dest)
    print(f"  完成! 文件大小: {os.path.getsize(dest)/1e6:.1f} MB")
    print(f"  SHA256: {h}")

    if args.verify:
        if h != args.verify.lower():
            print(f"  ✗ SHA256 不匹配! 期望 {args.verify}")
            sys.exit(1)
        print("  ✓ SHA256 校验通过")


if __name__ == "__main__":
    main()
