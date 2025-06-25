from cells import *


class Endstop_cell:
    def __init__(
            self, 
            s_cell_width, 
            s_cell_height, 
            esc_angle,
            c_cell_angle,
            c_cell_overlap,
            num_c_cells,
            c_cell_width = None,
            c_cell_height = None,
        ):
        self.s_cell_width = s_cell_width
        self.s_cell_height = s_cell_height
        self.esc_angle = esc_angle
        self.c_cell_angle = c_cell_angle
        self.c_cell_overlap = c_cell_overlap
        self.num_c_cells = num_c_cells
        self.c_cell_width = c_cell_width
        self.c_cell_height = c_cell_height

        self.s_cell = SCell(s_cell_width, s_cell_height, esc_angle)
        self.c_cells = CCells(self.num_c_cells, self.c_cell_overlap, self.c_cell_angle , self.s_cell, self.c_cell_width, self.c_cell_height)
        

        

    def get_response(self, image, x0, y0):
        self.s_cell.get_response(image, x0, y0)
        return
    