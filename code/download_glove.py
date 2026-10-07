"""下载并解压 GloVe 100 维词向量（仅解压 glove.6B.100d.txt）。

为什么用 Python 而不是 curl：
    Python 的 urllib 会自动读取 **Windows 系统代理**（与 pip 行为一致），
    而 curl.exe 默认不读系统代理，直连国外站点会被限速到 ~12 KB/s（约 19 小时）。
    curl 手动加 -x 也不稳定，所以统一用 Python。

用法（在项目根目录执行）：
    python code/download_glove.py
    python code/download_glove.py --keep_zip          # 保留下载的 zip
    python code/download_glove.py --member glove.6B.50d.txt   # 换别的维度

默认行为：下载 glove.6B.zip（约 822 MB）-> 只解压 glove.6B.100d.txt（约 347 MB）-> 删除 zip
"""
import argparse
import os
import time
import urllib.request
import zipfile

URL = "http://nlp.stanford.edu/data/glove.6B.zip"


def human(nbytes):
    return f"{nbytes / 1048576:.1f} MB"


def download(url, dst):
    """流式下载并按 5% 粒度打印进度、速度与 ETA。"""
    t0 = time.time()
    last_pct = -5.0
    last_mb_reported = 0.0
    done = 0

    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        print(f"文件大小: {human(total) if total else '未知'}")
        with open(dst, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)

                elapsed = max(time.time() - t0, 1e-6)
                speed = done / elapsed
                if total:
                    pct = done / total * 100
                    if pct - last_pct >= 5:
                        last_pct = pct
                        eta = (total - done) / speed
                        print(
                            f"  {pct:5.1f}%  {human(done)}/{human(total)}  "
                            f"{speed / 1048576:5.1f} MB/s  ETA {eta / 60:4.1f} min",
                            flush=True,
                        )
                elif done / 1048576 - last_mb_reported >= 50:
                    # 服务器没给 Content-Length 时按体积报进度
                    last_mb_reported = done / 1048576
                    print(f"  {human(done)}  {speed / 1048576:.1f} MB/s", flush=True)

    return done, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out_dir", default="embeddings", help="输出目录（相对项目根目录）")
    ap.add_argument(
        "--member",
        default="glove.6B.100d.txt",
        help="zip 内要解压的文件；glove.6B.zip 内含 50d/100d/200d/300d",
    )
    ap.add_argument("--keep_zip", action="store_true", help="保留 zip 不删除")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = os.path.join(args.out_dir, os.path.basename(args.url))
    target = os.path.join(args.out_dir, args.member)

    if os.path.exists(target):
        print(f"已存在，跳过：{target}（{human(os.path.getsize(target))}）")
        return

    print(f"下载: {args.url}")
    done, total = download(args.url, zip_path)
    print(f"下载完成: {human(done)}")
    if total and done != total:
        print(f"[warn] 实际下载字节数与 Content-Length 不一致，zip 可能不完整")
        return

    print(f"解压 {args.member} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(args.member, args.out_dir)
    print(f"OK  {target}  {human(os.path.getsize(target))}")

    if not args.keep_zip:
        os.remove(zip_path)
        print("已删除 zip 包")


if __name__ == "__main__":
    main()
