from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STYLE_ROOT = Path(__file__).resolve().parent.parent / "soltaninejad_style"
sys.path.append(str(STYLE_ROOT))

from metrics import dice_loss_from_logits, full_volume_metrics
from unet_model import UNet3D, count_parameters


def run(device_name: str) -> None:
    device = torch.device(device_name)
    model = UNet3D(base_channels=4, levels=3, norm="instance").to(device)
    model.eval()
    x = torch.rand((1, 1, 32, 32, 32), device=device)
    target = (x > 0.65).float()
    with torch.inference_mode():
        logits = model(x).logits
        loss = F.binary_cross_entropy_with_logits(logits, target) + dice_loss_from_logits(logits, target)

    assert tuple(logits.shape) == (1, 1, 32, 32, 32)
    assert torch.isfinite(loss).item()

    anisotropic = UNet3D(base_channels=2, levels=3, norm="instance").to(device)
    anisotropic.eval()
    x_aniso = torch.rand((1, 1, 16, 32, 32), device=device)
    with torch.inference_mode():
        logits_aniso = anisotropic(x_aniso).logits
    assert tuple(logits_aniso.shape) == (1, 1, 16, 32, 32)

    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[4:12, 4:12, 4:12] = 1
    metrics = full_volume_metrics(label, label)
    assert metrics["dice"] == 1.0
    assert metrics["cldice"] == 1.0

    print("3D U-Net baseline smoke test passed")
    print(f"device={device}")
    print(f"parameters={count_parameters(model)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"anisotropic_logits_shape={tuple(logits_aniso.shape)}")
    print(f"loss={float(loss.detach().cpu()):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-independent smoke test for the Plant CT 3D U-Net baseline.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    run(args.device)


if __name__ == "__main__":
    main()
