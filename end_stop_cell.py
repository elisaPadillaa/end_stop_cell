from cells import *
from funcs import rectification_func


class EndstopCell:
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
            # s_cell = None,
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

        # if s_cell == None:
        self.s_cell = SCell(s_cell_width, s_cell_height, esc_angle)
        # else: self.s_cell = s_cell
        
        self.c_cells = CCells(self.num_c_cells, self.c_cell_overlap, self.c_cell_angle , self.s_cell, self.c_cell_width, self.c_cell_height)
        

    def plot_points(self, x0, y0, image):
        centers = self.c_cells.get_centers(x0, y0, image)
        
        a = np.array(centers).reshape(-1,2)
        b = np.array(self.s_cell.plot_points(x0, y0)).reshape(-1,2)

        combine = np.concatenate([a, b], axis = 0)
        return combine
    
    def get_response(self, image, x0, y0, s_cell_resp=None):
        s_cell_gain, cL_cell_gain, cR_cell_gain = self.gains

        if s_cell_resp is None:
            s_cell_resp = self.s_cell.get_response(image, x0, y0)

        s_cell_resp = funcs.rectification_func(s_cell_resp)
        c_cell_respL, c_cell_respR = self.c_cells.get_response(image, x0, y0)
        esc_resp = s_cell_gain * s_cell_resp - (cL_cell_gain * c_cell_respL + cR_cell_gain * c_cell_respR)

        # if(x0 == 99 and y0== 55):
        #     print(f"{s_cell_resp} - ({c_cell_respL} + {c_cell_respR})")
        return esc_resp

        
class DegreeCurveESCell(EndstopCell):
    def __init__(
            self, 
            s_cell_width, 
            s_cell_height, 
            esc_angle, 
            c_cell_overlap, 
            num_c_cells,  
            gains,
            scaling_param = 1, 
            gamma = 0.01,
            c_cell_angle = 0,
            c_cell_width=None, 
            c_cell_height=None,
            # s_cell = None
        ):
        super().__init__(
            s_cell_width, 
            s_cell_height, 
            esc_angle,
            c_cell_overlap, 
            num_c_cells, 
            gains, 
            c_cell_angle, 
            c_cell_width, 
            c_cell_height,
            # s_cell
        )

        self.scaling_param = scaling_param
        self.gamma = gamma

    def plot_points(self, x0, y0, image):
        return super().plot_points(x0, y0, image)
    
    def get_response(self, image, x0, y0, s_cell_resp = None):
        esc_resp = super().get_response(image, x0, y0, s_cell_resp)
        return self.rectification_func(esc_resp)
    
    def rectification_func(self, resp):
        scaling = self.scaling_param
        return resp
        return (1 - math.e ** (-resp / scaling)) / 1 + (1 / self.gamma * math.e ** (-resp / scaling))
    
class SignCurveESCell(EndstopCell):
    def __init__(
            self, 
            s_cell_width, 
            s_cell_height, 
            esc_angle, 
            c_cell_overlap, 
            num_c_cells, 
            gains,
            c_cell_angle = 45, 
            c_cell_width=None, 
            c_cell_height=None,
            # s_cell = None
        ):

        super().__init__(
            s_cell_width, 
            s_cell_height, 
            esc_angle,
            c_cell_overlap, 
            num_c_cells, 
            gains, 
            c_cell_angle, 
            c_cell_width, 
            c_cell_height,
            # s_cell
        )

        self.pos_esc = EndstopCell(
            self.s_cell_width, 
            self.s_cell_height,  
            self.esc_angle,
            self.c_cell_overlap, 
            self.num_c_cells, 
            self.gains, 
            self.c_cell_angle,
            self.c_cell_width, 
            self.c_cell_height,
            # self.s_cell
        )

        self.neg_esc = EndstopCell(
            self.s_cell_width, 
            self.s_cell_height,
            self.esc_angle + 180, 
            self.c_cell_overlap, 
            self.num_c_cells,
            self.gains ,
            self.c_cell_angle, 
            self.c_cell_width, 
            self.c_cell_height,
            # self.s_cell
        )

    def plot_points(self, x0, y0, image):
        pos_points = self.pos_esc.plot_points(x0, y0, image)
        neg_points = self.neg_esc.plot_points(x0, y0, image)

        a = np.array(pos_points).reshape(-1,2)
        b = np.array(neg_points).reshape(-1,2)

        combine = np.concatenate([a, b], axis = 0)
        return a, b
    
    def get_response(self, image, x0, y0, s_cell_resp = None):
        pos_esc_resp = funcs.rectification_func(self.pos_esc.get_response(image, x0, y0, s_cell_resp))
        neg_esc_resp = funcs.rectification_func(self.neg_esc.get_response(image, x0, y0, s_cell_resp))
        
        return pos_esc_resp, neg_esc_resp
    