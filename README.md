# RELP

## Windows 安装

使用 Python 3.12。Windows 下的 Detectron2 来自相邻的
`../detectron_all` 目录（包含 Windows / CUDA 编译修正），以 editable 方式安装到
RELP 自己的 `.venv`。该目录需要持续保留；`detectron_all/.win` 是另一个独立环境，
不会自动被 RELP 使用。Windows 使用 PyTorch `2.14.0+cu132` 和 torchvision
`0.29.0+cu132`，需要匹配的 CUDA 13.2 Toolkit 才能重新编译原生扩展。

首次安装或重新编译时，在 RELP 根目录的 PowerShell 中执行：

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsInstall = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
& "$vsInstall\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:DISTUTILS_USE_SDK = '1'
$env:MAX_JOBS = '2'

# 配置变更后，或本地尚无 uv.lock 时，先更新锁文件。
uv lock
uv sync --locked --inexact
```

锁文件已经与配置一致后，后续只需 `uv sync --locked --inexact`；需要重新编译时，
仍应先初始化上面的编译环境。`--inexact` 保留额外安装的包，但不会跳过项目要求的
Detectron2，也不会借用另一个虚拟环境的安装。

验证 RELP 环境及 GPU 扩展：

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; from detectron2 import _C; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), _C.__file__)"
.\.venv\Scripts\python.exe -c "import torch; from detectron2.layers import nms_rotated; b=torch.tensor([[0.,0.,10.,10.,0.],[0.,0.,10.,10.,0.]],device='cuda'); s=torch.tensor([.9,.8],device='cuda'); print(nms_rotated(b,s,.5))"
```

第二条应返回只保留索引 `0` 的 CUDA tensor。它验证 Detectron2 的 CUDA 算子可用，
不代表已经验证完整 RELP 模型推理流水线。

Linux / macOS 的 Detectron2 继续使用 `pyproject.toml` 中固定的 GitHub commit。
