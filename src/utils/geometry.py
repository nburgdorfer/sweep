
from torch import Tensor
from typing import Tuple

import torch


def z_planes_from_disp(
    Z: Tensor, b: Tensor, f: Tensor, delta: float
) -> Tuple[Tensor, Tensor]:
    """Computes the near and far Z planes corresponding to 'delta' disparity steps between two cameras.

    Parameters:
        Z: Z buffer storing D depth plane hypotheses [B x C x D x H x W]. (shape resembles a typical PSV).
        b: The baseline between cameras [B].
        f: The focal length of camera [B].
        delta: The disparity delta for the near and far planes.

    Returns:
        The tuple of near and far Z planes corresponding to 'delta' disparity steps.
    """
    B, C, D, H, W = Z.shape
    b = b.reshape(B, 1, 1, 1, 1).repeat(1, C, D, H, W)
    f = f.reshape(B, 1, 1, 1, 1).repeat(1, C, D, H, W)

    near = Z * (b * f / ((b * f) + (delta * Z)))
    far = Z * (b * f / ((b * f) - (delta * Z)))

    return (near, far)

def disparity_hypothesis_planes(
    z: Tensor, b: Tensor, f: Tensor, delta: float, num_planes: int
) -> Tensor:
    """Computes the hypothesis z-planes corresponding to 'delta' disparity steps between two cameras from a central z plane.

    Parameters:
        z: Z buffer storing the central plane for each pixel [B x H x W].
        b: The baseline between cameras [B].
        f: The focal length of camera [B].
        delta: The disparity delta for the near and far planes.
        num_planes: The number of planes

    Returns:
        The hypothesis volume corresponding to 'delta' disparity increments.
    """
    B, H, W = z.shape
    b = b.reshape(B, 1, 1).repeat(1, H, W)
    f = f.reshape(B, 1, 1).repeat(1, H, W)

    hypotheses = torch.zeros((B, num_planes, H, W)).to(z)
    hypotheses[:,int(num_planes//2)] = z

    for p in range((num_planes//2)-1, -1, -1):
        hypotheses[:,p] = hypotheses[:,p+1] * (b * f / ((b * f) + (delta * hypotheses[:,p+1])))

    for p in range((num_planes//2)+1, num_planes):
        hypotheses[:,p] = hypotheses[:,p-1] * (b * f / ((b * f) - (delta * hypotheses[:,p-1])))
        
    return hypotheses

def get_disparity_planes(
    z_near: float, z_far: float, b: float, f: float, delta: float
) -> list[float]:
    """Computes the number of planes required to span [z_near -> z_far] planes (inclusive) with delta disparity.

    Parameters:
        z_near: The near Z plane.
        z_far: The far Z plane.
        b: The baseline between cameras [B].
        f: The focal length of camera [B].
        delta: The disparity delta for the near and far planes.

    Returns:
        The hypothesis planes corresponding to 'delta' disparity steps from z_near to z_far.
    """
    hypothesis_planes = [z_near]
    while hypothesis_planes[-1] <= z_far:
        current_plane = hypothesis_planes[-1] * (b * f / ((b * f) - (delta * hypothesis_planes[-1])))
        hypothesis_planes.append(current_plane)
    return hypothesis_planes

    