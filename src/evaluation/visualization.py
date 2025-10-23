from torch import Tensor
import numpy as np

def write_ply(filename: str, points: Tensor, colors: Tensor | None = None):
    if colors is not None:
        assert points.shape[0] == colors.shape[0]
        if (len(colors.shape) == 1):
            colors = colors.reshape(-1,1).repeat(1,3)
        elif (colors.shape[1] == 1):
            colors = colors.repeat(1,3)
        
        colors_np = colors.detach().cpu().numpy()
        colors_np = (colors_np - colors_np.min()) / 2.0
        colors_np = np.clip(colors_np, 0.0, 1.0)
    else:
        colors_np = None

    points_np = points.detach().cpu().numpy()

    num_vertices = points_np.shape[0]

    with open(filename,'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors_np is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(points_np.shape[0]):
            f.write(f"{points_np[i,0]} {points_np[i,1]} {points_np[i,2]}")
            if colors_np is not None:
                f.write(f" {int(colors_np[i,0])*255} {int(colors_np[i,1])*255} {int(colors_np[i,2])*255}\n")
            else:
                f.write(f"\n")
        
        