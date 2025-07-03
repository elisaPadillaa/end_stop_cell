import torch


class ShapeCell():
    def __init__(
            self, 
            img,
            theta,
            s_cell_width, 
            s_cell_height, 
            c_cell_overlap,
            num_c_cells,
            gains,
            c_cell_angle,
            c_cell_width = None,
            c_cell_height = None,
        ):
        self.img = img
        self.theta = theta
        self.s_cell_width = s_cell_width
        self.s_cell_height = s_cell_height
        self.c_cell_angle = c_cell_angle
        self.c_cell_overlap = c_cell_overlap
        self.num_c_cells = num_c_cells
        self.gains = gains
        self.c_cell_width = c_cell_width
        self.c_cell_height = c_cell_height

    def get_image(self):
        return


        


