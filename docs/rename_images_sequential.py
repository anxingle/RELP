#!/usr/bin/env python3
"""将目录第一层的图片连续编号，并保存 UTF-8 BOM 文件名映射 CSV。

从 RELP 根目录手动执行（无需安装额外依赖）：
    # 预览，不修改图片，也不生成 CSV：
    python3 docs/rename_images_sequential.py --dry-run

    # 正式重命名默认的 center_test_imprint 目录：
    python3 docs/rename_images_sequential.py

    # 指定其他目录和映射文件名（Windows 使用 python 代替 python3）：
    python3 docs/rename_images_sequential.py /path/to/images --csv filename_mapping.csv

Windows 路径也可作为目录参数传入 WSL，例如：
    python3 docs/rename_images_sequential.py 'C:\\Users\\anxin\\ssd_work\\Dinomaly\\data\\center_test_imprint' --dry-run

按原文件名自然排序（2 在 10 前），从 1 开始编号，保留原扩展名。
只处理目录第一层，不递归；不同扩展名的图片使用同一个编号序列。
默认在图片目录生成 filename_mapping.csv，包含 original_filename、new_filename。
已有 CSV 时拒绝执行，避免重复运行覆盖原始映射；--csv 相对路径以图片目录为基准。
采用临时目录中转，避免原目录已有 1.jpg、2.jpg 等文件时互相覆盖。
运行期间请勿让其他程序修改该目录；重命名不会改变图片内容。
"""

import argparse
import csv
import os
import re
from pathlib import Path, PureWindowsPath
from tempfile import mkdtemp


DEFAULT_DIRECTORY = r"/mnt/c/Users/anxin/ssd_work/Dinomaly/data/center_test_imprint"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def directory_path(value: str) -> Path:
    """Windows 原生路径直接使用；在 WSL 中转换盘符路径。"""
    windows_path = PureWindowsPath(value)
    if os.name != "nt" and windows_path.drive and windows_path.is_absolute():
        if len(windows_path.drive) != 2 or windows_path.drive[1] != ":":
            raise ValueError("WSL 下请使用 /mnt/... 路径，不支持自动转换 UNC 网络路径。")
        return Path("/mnt") / windows_path.drive[0].lower() / Path(*windows_path.parts[1:])
    return Path(value).expanduser()


def natural_key(path: Path) -> tuple:
    parts = tuple(int(part) if part.isdigit() else part.casefold()
                  for part in re.split(r"([0-9]+)", path.name))
    return parts, path.name


def rename_images(directory: Path, csv_path: Path, dry_run: bool = False) -> int:
    if not directory.is_dir():
        raise ValueError(f"图片目录不存在：{directory}")
    if csv_path.exists() or csv_path.is_symlink():
        raise ValueError(f"映射文件已存在，请保留它并检查是否已经重命名：{csv_path}")
    if csv_path.suffix.lower() != ".csv" or not csv_path.parent.is_dir():
        raise ValueError("映射文件须以 .csv 结尾，且其父目录必须存在。")

    images = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if not images:
        print(f"目录中没有支持的图片：{directory}")
        return 0

    plan = [(source, directory / f"{index}{source.suffix}")
            for index, source in enumerate(images, start=1)]
    sources = set(images)
    for source, target in plan:
        if (target.exists() or target.is_symlink()) and target not in sources:
            raise ValueError(f"目标路径已被其他文件或目录占用：{target}")
        print(f"{source.name} -> {target.name}")

    if dry_run:
        print(f"预览完成：共 {len(plan)} 张图片；未修改文件。映射将保存到：{csv_path}")
        return len(plan)

    # 先独占创建完整映射；即使进程意外中断，也保留文件名对应关系。
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_filename", "new_filename"])
        writer.writerows((source.name, target.name) for source, target in plan)

    # 中转名称使用原文件名，异常中断后可结合 CSV 识别尚未完成的文件。
    # 只有完成或成功回滚时才删除临时目录，避免清理尚未恢复的图片。
    staging = Path(mkdtemp(prefix=".rename_images_", dir=directory))
    staged = []
    completed = []
    try:
        for source, target in plan:
            temporary = staging / source.name
            source.rename(temporary)
            staged.append((source, temporary, target))
        for source, temporary, target in staged:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"执行期间目标路径被占用：{target}")
            temporary.rename(target)
            completed.append((source, temporary, target))
    except (Exception, KeyboardInterrupt):
        # 先撤回全部已完成目标，再恢复原名，避免数字文件名之间发生冲突。
        for source, temporary, target in reversed(completed):
            target.rename(temporary)
        for source, temporary, target in reversed(staged):
            if source.exists() or source.is_symlink():
                raise RuntimeError(f"无法恢复原名，请依据 {csv_path} 检查 {staging}：{source}")
            temporary.rename(source)
        staging.rmdir()
        csv_path.unlink()
        raise
    staging.rmdir()
    print(f"完成：已编号 {len(plan)} 张图片，映射保存到 {csv_path}")
    return len(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIRECTORY, help="图片目录（支持 Windows / WSL 路径）")
    parser.add_argument("--csv", type=Path, default=Path("filename_mapping.csv"), help="映射 CSV 路径，默认保存到图片目录")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不重命名或写入 CSV")
    args = parser.parse_args()
    try:
        directory = directory_path(args.directory).resolve()
        csv_path = args.csv.expanduser()
        if not csv_path.is_absolute():
            csv_path = directory / csv_path
        rename_images(directory, csv_path, args.dry_run)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
