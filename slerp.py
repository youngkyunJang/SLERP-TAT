import torch

def slerp(v0, v1, t):
    # Compute the cosine of the angle between the two vectors
    dot = (v0 * v1).sum(-1, keepdim=True)
    # If the dot product is close to 1, the vectors are nearly parallel
    # Use linear interpolation to avoid numerical precision issues
    close_condition = torch.abs(dot) > 0.9995
    linear_interp = v0 + t * (v1 - v0)
    linear_interp = linear_interp / linear_interp.norm(dim=-1, keepdim=True)
    # Compute the angle between the vectors and its sine
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    # Compute the scales for v0 and v1
    scale0 = torch.sin((1.0 - t) * theta) / sin_theta
    scale1 = torch.sin(t * theta) / sin_theta
    # Linearly interpolate between v0 and v1
    # This is equivalent to v0 * scale0 + v1 * scale1
    slerp_interp = scale0 * v0 + scale1 * v1
    # Normalize the output
    slerp_interp = slerp_interp / slerp_interp.norm(dim=-1, keepdim=True)
    return torch.where(close_condition, linear_interp, slerp_interp)
