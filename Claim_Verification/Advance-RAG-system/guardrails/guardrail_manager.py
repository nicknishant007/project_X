from .input_guard import InputGuard
from .output_guard import OutputGuard


class GuardrailManager:

    def __init__(self):

        self.input = InputGuard()

        self.output = OutputGuard()