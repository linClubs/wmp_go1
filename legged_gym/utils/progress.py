# Copyright (c) 2026-2027 zh
"""内容：为不提供进度回调的初始化操作显示等待动画和耗时。"""

import sys
import threading
from contextlib import contextmanager
from time import monotonic


@contextmanager
def waiting_progress(description):
    """显示不定长进度条；非终端输出只记录开始和结束。"""
    started = monotonic()
    stopped = threading.Event()
    interactive = sys.stdout.isatty()

    def update():
        position = 0
        while not stopped.wait(0.2):
            bar = [" "] * 24
            bar[position % len(bar)] = "#"
            print(f"\r{description} [{''.join(bar)}] {monotonic() - started:.1f}s", end="", flush=True)
            position += 1

    print(f"{description}: started", flush=True)
    worker = threading.Thread(target=update, daemon=True) if interactive else None
    if worker is not None:
        worker.start()
    status = "interrupted"
    try:
        yield
        status = "completed"
    finally:
        stopped.set()
        if worker is not None:
            worker.join()
        prefix = "\r\033[2K" if interactive else ""
        print(f"{prefix}{description}: {status} in {monotonic() - started:.1f}s", flush=True)
