import math
import numpy as np
import torch
import torch.nn.functional as F
import funcs


class SCell():
    gabor_params = {
        1: {"sigma": 1.482, "lambda": 4},
        2: {"sigma": 2.201, "lambda": 5.66},
        3: {"sigma": 3.128, "lambda": 8},  
        4: {"sigma": 4.467, "lambda": 11.314},
        5: {"sigma": 6.322, "lambda": 16},
        6: {"sigma": 8.971, "lambda": 22.327}
    }
    def __init__(self, cell_type, angle):
        self.width = SCell.gabor_params[cell_type]["sigma"] * 2
        self.height = SCell.gabor_params[cell_type]["lambda"] * 2
        self.angle = (270 - angle) % 360
        self.angle_raw = angle
        # print (self.angle)

    def get_response(self, image, x0, y0):
        # patch = self.get_patch(image, x0, y0) 
        # return patch.sum().item()
        # if y0== 86 and x0 == 39:
        #     print(image[int(x0), int(y0)])
        
        return image[int(y0), int(x0)]
    
    def get_patch(self, image, x0, y0):
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

        return patch
    
    def plot_points(self, x0, y0):
        points = [(x0, y0)] 
        offsets = [self.height / 2, -self.height / 2]
        for dy in offsets:
            x_rot, y_rot = funcs.rotate_point_around_center(x0, y0 + dy, x0, y0, self.angle_raw)
            points.append((x_rot, y_rot))
        return points


class CCells() :
    def __init__(
            self, 
            num_s_cells,
            overlap, 
            angle,
            s_cell,
            cell_type = None,
        ):
        self.num_s_cells = num_s_cells
        self.overlap = overlap
        self.angle = angle
        self.s_cell = s_cell

        if cell_type == None: 
            self.width = s_cell.width
        else: self.width = SCell.gabor_params[cell_type]["sigma"] * 2
        if cell_type == None: 
            self.height = s_cell.height
        else: self.height = SCell.gabor_params[cell_type]["lambda"] * 2

    def get_response(self, image, x0, y0):
        centers = self.get_centers(x0, y0, image)
        left_response = 0
        right_response = 0
        for i, group in enumerate(centers):     
            for x, y in group:  
                c_cell_response = self.s_cell.get_response(image, x, y)
                c_cell_response = funcs.rectification_func(c_cell_response)
                weight = self.calculate_weight()
                if i == 0: left_response += (c_cell_response * weight)
                else: right_response += (c_cell_response * weight)

        return left_response, right_response
    
    def calculate_weight(self):
        # complete
        return 1


    def get_centers(self, x0, y0, image):
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

            # Generate all the other centers around the initial point
            generation_angle = 90 + -sign * self.angle + s_cell_angle
            centers.append(self.generate_points((x_rot, y_rot), self.num_s_cells, self.width/2, generation_angle, image.shape))  
  
        return centers  # returns [left_center, right_center]
    

    def generate_points(self, center, num_points, distance, angle, image_shape):
        cx, cy = center
        angle_rad = math.radians(angle)
        max_y, max_x = image_shape  # Note: y = rows, x = cols

        # Calculate offset indices
        if num_points % 2 == 1:
            offset_range = range(-(num_points // 2), num_points // 2 + 1)
        else:
            half = num_points // 2
            offset_range = [i + 0.5 for i in range(-half, half)]

        points = []
        for offset in offset_range:
            dx = offset * distance * math.cos(angle_rad)
            dy = offset * distance * -math.sin(angle_rad)

            x = cx + dx
            y = cy + dy

            x_clamped = min(max(0, int(round(x))), max_x - 1)
            y_clamped = min(max(0, int(round(y))), max_y - 1)

            points.append((x_clamped, y_clamped))

        return points


    def add_distance(self, x, y, d, angle):
        angle_rad = math.radians(angle)  # convert to radians
        x_new = x + d * math.cos(angle_rad)
        y_new = y - d * math.sin(angle_rad)
        return x_new, y_new
    
    
    # def get_centers_double(self, x0, y0, angles):
    #     centers = []
    #     s_cell_angle = self.s_cell.angle
    #     d = (self.s_cell.height / 2) + (self.height / 2) - self.overlap
        
    #     # For both sides
    #     for sign in [(1), (-1)]:
    #         # Move center to the side
    #         x_center, y_center = self.add_distance(x0, y0, sign * d, s_cell_angle)

    #         # Get edge of simple cell
    #         x_s_edge, y_s_edge = self.add_distance(x0, y0, sign * (self.s_cell.height / 2), s_cell_angle)

    #         # Get edge of complex cell (backwards from center)
    #         x_c_edge, y_c_edge = self.add_distance(x_center, y_center, -sign * (self.height / 2), s_cell_angle)

    #         # Midpoint
    #         x_mid = (x_s_edge + x_c_edge) / 2
    #         y_mid = (y_s_edge + y_c_edge) / 2

    #         for angle in angles: #must be [angle, angle + 180]
    #             # Rotate around midpoint
    #             x_rot, y_rot = funcs.rotate_point_around_center(x_center, y_center, x_mid, y_mid, sign * angle)

    #             # Generate all the other centers around the initial point
    #             generation_angle = 90 + -sign * angle + s_cell_angle
    #             centers.append(self.generate_points((x_rot, y_rot), self.num_s_cells, self.width/2, generation_angle))  


       
    #     return centers  # returns [left_center, right_center]
    




        
        
