from end_stop_cell import *
from cells import SCell


class CurvatureCell():
    def __init__(
            self, 
            s_cell_width, 
            s_cell_height, 
            esc_angle,
            c_cell_overlap,
            num_c_cells,
            gains,
            c_cell_angle,
            c_cell_width = None,
            c_cell_height = None,
        ):
        self.s_cell_width = s_cell_width
        self.s_cell_height = s_cell_height
        self.esc_angle = esc_angle
        self.c_cell_angle = c_cell_angle
        self.c_cell_overlap = c_cell_overlap
        self.num_c_cells = num_c_cells
        self.gains = gains
        self.c_cell_width = c_cell_width
        self.c_cell_height = c_cell_height

        self.simple_cell = SCell(
            self.s_cell_width,
            self.s_cell_height,
            self.esc_angle
        )

        self.end_stopped_cell = DegreeCurveESCell(
            self.s_cell_width,
            self.s_cell_height, 
            self.esc_angle,
            self.c_cell_overlap,
            self.num_c_cells,
            self.gains,
            self.c_cell_angle,
            self.c_cell_width,
            self.c_cell_height,
            self.simple_cell,
        )

        self.sign_curve_cells = SignCurveESCell(
            self.s_cell_width,
            self.s_cell_height, 
            self.esc_angle,
            self.c_cell_overlap,
            self.num_c_cells,
            self.gains,
            self.c_cell_angle,
            self.c_cell_width,
            self.c_cell_height,
            self.simple_cell,
        )


    def get_response(self, image, x0, y0):
        s_cell_resp = self.simple_cell.get_response(image, x0, y0)
        sign_cell_pos_resp, sign_cell_neg_resp = self.sign_curve_cells.get_response(image, x0, y0, s_cell_resp)
        esc_resp = self.end_stopped_cell.get_response(image, x0, y0, s_cell_resp)
        pos_resp = []
        neg_resp = []

        if sign_cell_pos_resp > sign_cell_neg_resp:
            pos_resp.append((x0, y0, esc_resp))
        elif sign_cell_pos_resp < sign_cell_neg_resp:
            neg_resp.append((x0, y0, esc_resp))

        return pos_resp, neg_resp
        