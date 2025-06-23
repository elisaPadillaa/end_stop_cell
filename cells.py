import numpy as np
import torch
import torch.nn.functional as F


class SCell():
    def __init__(self, width, height, angle):
        self.width = width
        self.height = height
        self.angle = angle

    def get_response(self, image, x0, y0):
        """
        image: torch tensor [H, W] (filtered image)
        center: (x, y) position in pixels (float or int)
        angle_deg: 0, 45, 90, 135, etc.
        size: (h, w) of the rectangle patch (in pixels)
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        img_h, img_w = image.shape 
        h = self.height
        w = self.width

        # Normalized coordinates: from -1 to 1
        tx = (2 * x0 / (img_w - 1)) - 1
        ty = (2 * y0 / (img_h - 1)) - 1
        theta_rad = np.deg2rad(self.angle)
        cos_a = np.cos(theta_rad)
        sin_a = np.sin(theta_rad)

        theta = torch.tensor([[
            [cos_a * (w / img_w), -sin_a * (h / img_h), tx],
            [sin_a * (w / img_w),  cos_a * (h / img_h), ty]
        ]], dtype=torch.float32)  # Shape (1, 2, 3)

        grid = F.affine_grid(theta, size=(1, 1, h, w), align_corners=True)
        # patch = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=True)[0, 0]
        patch = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode='nearest')[0, 0]

        return patch.sum()


class CCell() :
    def __init__(self, n_simple_cells):
        pass