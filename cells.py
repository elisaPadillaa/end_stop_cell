import math
import numpy as np
import torch
import torch.nn.functional as F
import funcs


class SCell():
    def __init__(self, width, height, angle):
        self.width = width
        self.height = height
        self.angle = (270 - angle) % 360
        self.angle_raw = angle
        # print (self.angle)

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
        theta_rad = np.deg2rad(self.angle_raw)
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
            num_s_cells,
            overlap, 
            angle,
            s_cell,
            width = None,
            height = None,
        ):
        self.num_s_cells = num_s_cells
        self.overlap = overlap
        self.angle = angle
        self.s_cell = s_cell

        if width == None: 
            self.width = s_cell.width
        else: self.width = width
        if height == None: 
            self.height = s_cell.height
        else: self.height = height

    def get_response(self, image, x0, y0):
        centers = self.get_centers(x0, y0)

        return
    
    def get_centers(self, x0, y0):
        centers_main = []
        centers = []
        s_cell_angle = self.s_cell.angle
        d = (self.s_cell.height / 2) + (self.height / 2) - self.overlap
        
        # For both sides
        for sign in [(1), (-1)]:
            # Move center to the side
            x_center, y_center = self.add_distance(x0, y0, sign * d, s_cell_angle)

            # Get edge of simple cell
            x_s_edge, y_s_edge = self.add_distance(x0, y0, sign * (self.s_cell.height / 2), s_cell_angle)

            # Get edge of complex cell (backwards from center)
            x_c_edge, y_c_edge = self.add_distance(x_center, y_center, -sign * (self.height / 2), s_cell_angle)

            # Midpoint
            x_mid = (x_s_edge + x_c_edge) / 2
            y_mid = (y_s_edge + y_c_edge) / 2

            # Rotate around midpoint
            x_rot, y_rot = funcs.rotate_point_around_center(x_center, y_center, x_mid, y_mid, sign * self.angle)

            # centers.append((x_center, y_center))
            centers_main.append((x_rot, y_rot))

        for i, points in enumerate(centers_main):
            sign = 1 if i == 0 else -1
            centers.append(self.generate_points(points, self.num_s_cells, self.width/2,  s_cell_angle + sign * self.angle))  

        return centers  # returns [left_center, right_center]

    def generate_points(self, center, num_points, distance, angle):
        cx, cy = center
        angle_rad = math.radians(angle)
        
        # Calculate offset indices
        if num_points % 2 == 1:
            # Odd: symmetric around center, includes (0, 0)
            offset_range = range(-(num_points // 2), num_points // 2 + 1)
        else:
            # Even: symmetric, but no point exactly at center
            half = num_points // 2
            offset_range = [i + 0.5 for i in range(-half, half)]
        
        points = []
        for offset in offset_range:
            dx = offset * distance * math.cos(angle_rad)
            dy = offset * distance * -math.sin(angle_rad)
            points.append((cx + dx, cy + dy))

        return points


    def add_distance(self, x, y, d, angle):
        angle_rad = math.radians(angle)  # convert to radians
        x_new = x + d * math.cos(angle_rad)
        y_new = y - d * math.sin(angle_rad)
        return x_new, y_new
    




        
        
