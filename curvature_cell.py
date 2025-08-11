
import torch

from cells import SCell
from end_stop_cell import DegreeCurveESCell, SignCurveESCell




class CurvatureCell():
    def __init__(
            self, 
            s_cell_type, 
            esc_angle,
            c_cell_overlap,
            num_c_cells,
            gains,
            device
        ):
        self.s_cell_type = s_cell_type
        self.esc_angle = esc_angle
        self.c_cell_overlap = c_cell_overlap
        self.num_c_cells = num_c_cells
        self.gains = gains
        self.device = device

        self.simple_cell = SCell(
            self.s_cell_type,
            self.esc_angle,
            self.device,
        )

        self.end_stopped_cell = DegreeCurveESCell(
            self.s_cell_type, 
            self.esc_angle,
            self.c_cell_overlap,
            self.num_c_cells,
            self.gains,
            self.device,
        )

        self.sign_curve_cells = SignCurveESCell(
            self.s_cell_type, 
            self.esc_angle,
            self.c_cell_overlap,
            self.num_c_cells,
            self.gains,
            self.device,
        )


    def get_response(self, images):
        #Generate the simple cell response only one time
        s_cell_resp = self.simple_cell.get_response(images)
        # Positive and negative sign cell response
        sign_cell_pos_resp, sign_cell_neg_resp = self.sign_curve_cells.get_response(images, s_cell_resp)
        # Degree of curve cell response
        esc_resp = self.end_stopped_cell.get_response(images, s_cell_resp)
        # Any negative value = 0
        esc_resp = torch.clamp(esc_resp, min = 0)

        pos_resp = torch.zeros_like(esc_resp)
        neg_resp = torch.zeros_like(esc_resp)

        mask_pos = sign_cell_pos_resp > sign_cell_neg_resp
        mask_neg = sign_cell_pos_resp < sign_cell_neg_resp

        pos_resp[mask_pos] = esc_resp[mask_pos]
        neg_resp[mask_neg] = esc_resp[mask_neg]

        return pos_resp, neg_resp
        