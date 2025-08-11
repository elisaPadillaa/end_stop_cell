import math
import numpy as np
import torch
import torch.nn.functional as F

import funcs

#This is the cell
class SCell():
    # For each gabor size 1 to 6 the cell size depends on sigma (width = sigma * 2) and lambda (height = lambda * 2)
    gabor_params = {
        1: {"sigma": 1.482, "lambda": 4},
        2: {"sigma": 2.201, "lambda": 5.66},
        3: {"sigma": 3.128, "lambda": 8},  
        4: {"sigma": 4.467, "lambda": 11.314},
        5: {"sigma": 6.322, "lambda": 16},
        6: {"sigma": 8.971, "lambda": 22.327}
    }
    def __init__(self, cell_type, angle, device):
        self.width = SCell.gabor_params[cell_type]["sigma"] * 2
        self.height = SCell.gabor_params[cell_type]["lambda"] * 2
        # Adjust angle to match internal coordinate system
        # Gabor angle system defines 0º at the bottom and goes clockwise while computer's coordinate system 
        # defines 0° pointing to the right (east) and increases counterclockwise. 
        # This tansformation converts from our angle system to the one used by the computer.
        self.angle = (270 - angle) % 360
        # Store the original "raw" angle for certain processes
        self.angle_raw = angle
        # Cell_type means a number from 1-6 representing the gabor filter size which is the cell size
        self.cell_type = cell_type
        self.device = device

        # Reference of gabor image for each orientation (0º = imaga[0])
        switch = {
            0: 0,
            45: 1,
            90: 2,
            135: 3
        }

        self.angle_class = switch.get(self.wrap_angle(self.angle_raw), -1)
        if self.angle_class == -1:
            raise Exception("Angle must be 0º, 45º, 90º or 135º")
        
    
    def wrap_angle(self, angle_raw):
        angle = angle_raw % 360       
        return angle if angle < 180 else angle - 180

    # Return the gabor image in the cell orientation
    def get_response(self, images):
        image = images[self.angle_class]
        return image
    


class CCells() :
    def __init__(
            self, 
            num_s_cells,
            overlap, 
            angle,
            s_cell,
        ):
        self.num_s_cells = num_s_cells
        self.overlap = overlap
        self.angle = angle
        self.s_cell = s_cell
        self.width = s_cell.width
        self.height = s_cell.height
        self.device = s_cell.device
        # I create two simple cells so that each side left and right has its own angle
        self.s_cell_left = SCell(self.s_cell.cell_type, self.s_cell.angle_raw + self.angle, self.device)
        self.s_cell_right = SCell(self.s_cell.cell_type, self.s_cell.angle_raw + self.angle * (-1), self.device)
        

    def get_response(self, images, points):

        #Two times: one for the left simple cells and one for the right simple cells
        for i, group in enumerate(points):   
            # Decide which cell to use
            if i == 0: cell = self.s_cell_left
            else: cell = self.s_cell_right
           
            # Return image in the orientation of the simple cell
            main_img = cell.get_response(images)
            # Initialize left and right response
            if i == 0: left_response = torch.zeros_like(main_img)
            else: right_response = torch.zeros_like(main_img)
            
            # Loops for every simple cell in the complex cell (5 times = 5 simple cells in the complex cell)
            for a, b in group:  
                # Shift the image by adding the relative position of the simple cell from the complex cell (look at paper for example or ask Elisa)
                c_cell_response = self.shift_img(main_img, a, b)
                # Rectification function all negative values turn into 0
                c_cell_response = torch.clamp(c_cell_response, min = 0)
                weight = self.calculate_weight() #NO WEIGHT NOW MUST COMPLETE
                # Add to the left or right response
                if i == 0: left_response += (c_cell_response * weight)
                else: right_response += (c_cell_response * weight)

        return left_response, right_response
    
    def shift_img(self, image, a, b):
       
        image = image.unsqueeze(0).unsqueeze(0)
        
        theta = torch.tensor([[
            [1, 0, a / (image.shape[3] / 2)],  # x shift (width)
            [0, 1, b / (image.shape[2] / 2)]   # y shift (height)
        ]], dtype=torch.float)

        grid = F.affine_grid(theta, image.size(), align_corners=False).to(self.device)
        shifted_image = F.grid_sample(image, grid, padding_mode="zeros", align_corners=False)

        # Remove batch dimension
        shifted_image = shifted_image.squeeze(0).squeeze(0)
        return shifted_image
    
    def calculate_weight(self):
        # complete
        return 1

    def get_centers(self):
        offsets = []
        s_cell_angle = self.s_cell.angle
        d = (self.s_cell.height / 2) + (self.height / 2) - self.overlap

        for sign in [1, -1]:
            # Step 1: Move center to the side (relative to origin)
            x_center_offset, y_center_offset = self.add_distance(0, 0, sign * d, s_cell_angle)

            # Step 2: Get edges (still relative)
            x_s_edge, y_s_edge = self.add_distance(0, 0, sign * (self.s_cell.height / 2), s_cell_angle)
            x_c_edge, y_c_edge = self.add_distance(x_center_offset, y_center_offset, -sign * (self.height / 2), s_cell_angle)

            # Step 3: Midpoint (relative)
            x_mid = (x_s_edge + x_c_edge) / 2
            y_mid = (y_s_edge + y_c_edge) / 2

            # Step 4: Rotate around midpoint (relative)
            x_rot, y_rot = funcs.rotate_point_around_center(
                x_center_offset, y_center_offset,
                x_mid, y_mid,
                sign * self.angle
            )

            # Step 5: Generate offsets from rotated point
            generation_angle = 90 + -sign * self.angle + s_cell_angle
            offsets.append(self.generate_relative_offsets((x_rot, y_rot), self.num_s_cells, self.width / 2, generation_angle))

        return offsets  # returns [left_offsets, right_offsets]
    
    def generate_relative_offsets(self, center_offset, num_points, distance, angle):
        cx, cy = center_offset
        angle_rad = math.radians(angle)

        if num_points % 2 == 1:
            offset_range = range(-(num_points // 2), num_points // 2 + 1)
        else:
            half = num_points // 2
            offset_range = [i + 0.5 for i in range(-half, half)]

        offsets = []
        for offset in offset_range:
            dx = offset * distance * math.cos(angle_rad)
            dy = offset * distance * -math.sin(angle_rad)

            total_dx = cx + dx
            total_dy = cy + dy

            offsets.append((total_dx, total_dy))  # still floats for later precision

        return offsets

    def add_distance(self, x, y, d, angle):
        angle_rad = math.radians(angle)  # convert to radians
        x_new = x + d * math.cos(angle_rad)
        y_new = y - d * math.sin(angle_rad)
        return x_new, y_new
    

        
