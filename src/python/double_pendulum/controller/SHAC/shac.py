from double_pendulum.controller.abstract_controller import AbstractController


class SHAC(AbstractController):
    def __init__(self):
        super().__init__()

    def get_control_output(self, x, t=None):
        return super().get_control_output(x, t)
