#!/usr/bin/env python3
"""把本地文件推送到 GitHub 仓库（走 Git Data API）。

为什么不用 git push：某些沙箱环境会拦截 .git 目录下的 rename/unlink，
导致 `git add/commit` 失败。本脚本直接调用 GitHub API 建 blob → tree → commit → 更新 ref，
不碰本地 .git，因此不受影响。

用法：
    python tools/push_github.py "提交信息"
    python tools/push_github.py "提交信息" --repo owner/name --branch main
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"

# 需要推送的文件（相对项目根目录）
DEFAULT_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "run.py",
    "web/index.html",
    "web/app.js",
    "web/styles.css",
    "web/vendor/echarts.min.js",
    "tools/push_github.py",
]


def collect(root: str) -> list[str]:
    files = list(DEFAULT_FILES)
    qbt = os.path.join(root, "qbt")
    if os.path.isdir(qbt):
        files += [os.path.join("qbt", f) for f in sorted(os.listdir(qbt)) if f.endswith(".py")]
    return [f for f in files if os.path.isfile(os.path.join(root, f))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    ap.add_argument("--repo", default="2422725977/quantlab")
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    owner, _, repo = args.repo.partition("/")

    try:
        token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                               timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        token = ""
    if not token:
        print("无法获取 gh token，请先 `gh auth login`", file=sys.stderr)
        return 1

    def req(method: str, path: str, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        r = urllib.request.Request(
            API + path, data=data, method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "quantlab-push",
            },
        )
        try:
            return json.loads(urllib.request.urlopen(r, timeout=180).read().decode())
        except urllib.error.HTTPError as e:
            return {"_error": e.code, "_body": e.read().decode()[:300]}

    # 当前分支的 commit（作为 parent）
    ref = req("GET", f"/repos/{owner}/{repo}/git/ref/heads/{args.branch}")
    parent = None
    if "object" in ref:
        parent = ref["object"]["sha"]
        print(f"当前 {args.branch}: {parent[:12]}")
    else:
        print(f"分支不存在，将创建初始提交：{ref}")

    files = collect(root)
    tree = []
    for f in files:
        raw = open(os.path.join(root, f), "rb").read()
        b = req("POST", f"/repos/{owner}/{repo}/git/blobs",
                {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
        if "sha" not in b:
            print(f"  ✗ {f}: {b}", file=sys.stderr)
            continue
        tree.append({"path": f.replace(os.sep, "/"), "mode": "100644", "type": "blob", "sha": b["sha"]})
        print(f"  ✓ {f}  ({len(raw):,} bytes)")

    t = req("POST", f"/repos/{owner}/{repo}/git/trees", {"tree": tree})
    if "sha" not in t:
        print(f"建树失败：{t}", file=sys.stderr)
        return 1

    payload: dict = {"message": args.message, "tree": t["sha"]}
    if parent:
        payload["parents"] = [parent]
    c = req("POST", f"/repos/{owner}/{repo}/git/commits", payload)
    if "sha" not in c:
        print(f"建提交失败：{c}", file=sys.stderr)
        return 1
    print(f"\n提交 {c['sha'][:12]}  ({len(tree)} 个文件)")

    if parent:
        r = req("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/{args.branch}", {"sha": c["sha"]})
    else:
        r = req("POST", f"/repos/{owner}/{repo}/git/refs", {"ref": f"refs/heads/{args.branch}", "sha": c["sha"]})
    if "ref" not in r and "object" not in r:
        print(f"更新 ref 失败：{r}", file=sys.stderr)
        return 1
    print(f"\n完成 → https://github.com/{owner}/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
