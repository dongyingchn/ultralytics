# name=measure_flops_thop.py
import torch
import numpy as np
from ultralytics import YOLO

def measure_with_thop(model, input_size=(640,640), device='cpu'):
    m = model.model if hasattr(model, 'model') else model  # get nn.Module
    # del m.model
    m.eval().to(device)

    N = 1
    C = 3
    H, W = input_size
    inp = torch.randn(N, C, H, W, device=device)

    try:
        from thop import profile
    except Exception as e:
        raise RuntimeError("Please install thop: pip install thop") from e

    # profile may fail if model uses unsupported ops; wrap in try/except
    try:
        macs, params = profile(m, inputs=(inp,), verbose=False)
    except Exception as e:
        raise RuntimeError(f"thop profiling failed: {e}")

    flops = 2 * macs  # approximate FLOPs
    print(f"Input: {N}x{C}x{H}x{W}")
    print(f"Params: {params:,} ({params/1e6:.3f} M)")
    print(f"MACs: {macs:,} ({macs/1e9:.3f} G)")
    print(f"FLOPs (approx): {flops:,} ({flops/1e9:.3f} G)")
    return flops, params

if __name__ == '__main__':
    # model_path = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/minieye-driving-d4q-roi4/weights/best.pt"
    # model = YOLO(model_path)

    # model = YOLO('yolo11s.yaml')  # or load a different model
    model = YOLO('/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/minieye-driving-d4q-yolo11s-full_image-all/weights/last.pt')
    measure_with_thop(model, input_size=(384,960), device='cuda' if torch.cuda.is_available() else 'cpu')