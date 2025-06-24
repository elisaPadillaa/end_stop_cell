import numpy as np
import torch
import torch.nn.functional as F
import math


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

        grid = F.affine_grid(theta, size=(1, 1, h, w), align_corners=True) #shape = (batch, channel, height, width)
        patch = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=True)[0, 0]

        return patch.sum(), patch


class CCells() :
    def __init__(
            self, 
            n_simple_cells,
            width,
            height,
            overlap, 
            angle,
            s_cell):
        self.n_simple_cells = n_simple_cells
        self.width = width
        self.height = height
        self.overlap = overlap
        self.angle = angle
        self.s_cell = s_cell


    def get_response(self, image, x0, y0):
        centers = self.get_centers(x0, y0)
        return
    
    def get_centers(self, x0, y0):
        angle = self.s_cell.angle
        d = (self.s_cell.width / 2) + (self.width / 2) - self.overlap
        x_center, y_center = self.add_distance(x0, y0, d, angle)
        d1 = self.s_cell.width / 2
        d2 = self.width / 2 * -1
        x1, y1 = self.add_distance(x0, y0, d1, angle)
        x2, y2 = self.add_distance(x_center, y_center, d2, angle)
        x_o = (x1 + x2) / 2
        y_o = (y1 + y2) / 2
        x_center, y_center = self.rotate_point_around_center(x_center, y_center, x_o, y_o, self.angle)
        return 

    def add_distance(self, x, y, d, angle):
        angle_rad = math.radians(angle)  # convert to radians
        x_new = x + d * math.cos(angle_rad)
        y_new = y + d * math.sin(angle_rad)
        return x_new, y_new
    
    def rotate_point_around_center(x, y, x0, y0, angle_deg):
        angle_rad = math.radians(angle_deg)
        dx = x - x0
        dy = y - y0
        x_rot = math.cos(angle_rad) * dx - math.sin(angle_rad) * dy + x0
        y_rot = math.sin(angle_rad) * dx + math.cos(angle_rad) * dy + y0
        return x_rot, y_rot




        
        
