import numpy as np
import torch

from cells import CCells, SCell



class EndstopCell:
    def __init__(
            self, 
            s_cell_type, 
            esc_angle,
            c_cell_overlap,
            num_c_cells,
            gains,
            c_cell_angle, # This angle is added on to the esc angle 
            device,
        ):
        self.s_cell_type = s_cell_type
        self.esc_angle = esc_angle
        self.c_cell_angle = c_cell_angle
        self.c_cell_overlap = c_cell_overlap
        self.num_c_cells = num_c_cells
        self.gains = gains
        self.device = device

        
        self.s_cell = SCell(s_cell_type, esc_angle, device)
        
        self.c_cells = CCells(self.num_c_cells, self.c_cell_overlap, self.c_cell_angle , self.s_cell)
        self.points = self.c_cells.get_centers()
        print()
    
    def get_response(self, images, s_cell_resp=None):
        s_cell_gain, cL_cell_gain, cR_cell_gain = self.gains

        if s_cell_resp is None:
            s_cell_resp = self.s_cell.get_response(images)

        s_cell_resp = torch.clamp(s_cell_resp, min = 0)
        c_cell_respL, c_cell_respR = self.c_cells.get_response(images, self.points)
        esc_resp = s_cell_gain * s_cell_resp - (cL_cell_gain * c_cell_respL + cR_cell_gain * c_cell_respR)

        return esc_resp

        
class DegreeCurveESCell(EndstopCell):
    def __init__(
            self, 
            s_cell_type, 
            esc_angle, 
            c_cell_overlap, 
            num_c_cells,  
            gains,
            device,
            scaling_param = 1, 
            gamma = 0.01,
            c_cell_angle = 0,
        ):
        super().__init__(
            s_cell_type, 
            esc_angle,
            c_cell_overlap, 
            num_c_cells, 
            gains, 
            c_cell_angle, 
            device,
        )

        self.scaling_param = scaling_param
        self.gamma = gamma

    def get_response(self, images, s_cell_resp = None):
        esc_resp = super().get_response(images, s_cell_resp)
        return self.rectification_func(esc_resp)
    
    def rectification_func(self, resp):
        scaling = self.scaling_param
        #COMPLETE use the actual function below
        return resp
        return (1 - math.e ** (-resp / scaling)) / 1 + (1 / self.gamma * math.e ** (-resp / scaling))
    
class SignCurveESCell(EndstopCell):
    def __init__(
            self, 
            s_cell_type, 
            esc_angle, 
            c_cell_overlap, 
            num_c_cells, 
            gains,
            device,
            c_cell_angle = 45, 
        ):

        super().__init__(
            s_cell_type, 
            esc_angle,
            c_cell_overlap, 
            num_c_cells, 
            gains, 
            c_cell_angle, 
            device
        )

        #Positive cell
        self.pos_esc = EndstopCell(
            self.s_cell_type,  
            self.esc_angle,
            self.c_cell_overlap, 
            self.num_c_cells, 
            self.gains, 
            self.c_cell_angle,
            self.device,
        )

        #Negative cell
        self.neg_esc = EndstopCell(
            self.s_cell_type,
            self.esc_angle + 180, 
            self.c_cell_overlap, 
            self.num_c_cells,
            self.gains ,
            self.c_cell_angle, 
            self.device,
        )
    
    def get_response(self, images, s_cell_resp = None):
        pos_esc_resp = torch.clamp(self.pos_esc.get_response(images, s_cell_resp), min = 0)
        neg_esc_resp = torch.clamp(self.neg_esc.get_response(images, s_cell_resp), min = 0)
        
        return pos_esc_resp, neg_esc_resp
    