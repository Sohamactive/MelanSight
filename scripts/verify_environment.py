"""Print Python/PyTorch/CUDA details and run a small computation on the GPU."""

import platform
import sys

import torch
import torchvision


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CUDA runtime (torch.version.cuda): {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA is unavailable; do not start GPU training.")
        return 1
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}")
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")
    product = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
    torch.cuda.synchronize()
    print(f"GPU matmul OK, result device={product.device}, finite={bool(torch.isfinite(product).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
