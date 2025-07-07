import torch


class ShapeCell():
    def __init__(
            self, 
            img,
            theta,
            s_cell_type, 
            c_cell_overlap,
            num_c_cells,
            gains,
            c_cell_angle,
            c_cell_type = None,
        ):
        self.img = img
        self.theta = theta
        self.s_cell_type = s_cell_type
        self.c_cell_angle = c_cell_angle
        self.c_cell_overlap = c_cell_overlap
        self.num_c_cells = num_c_cells
        self.gains = gains
        self.c_cell_type = c_cell_type

    def get_image(self):
        return


        


