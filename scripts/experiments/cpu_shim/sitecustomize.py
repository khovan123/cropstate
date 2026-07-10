"""Auto-loaded shim: neutralize the hanging CUDA probe on this broken-driver box.

When CROPSTATE_NO_CUDA=1, patch torch.cuda.is_available/device_count to return
CPU-only WITHOUT issuing the driver ioctl that hangs uninterruptibly here.
Importing torch itself is fine; only the CUDA availability probe hangs.
"""
import os
if os.environ.get("CROPSTATE_NO_CUDA") == "1":
    try:
        import torch
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
    except Exception:
        pass
