from .cnn import Net


class method_factory():
    def __init__(self, parse, split_layer):
        self.model = parse['model']
        self.split_layer = split_layer

    def factory(self):
        pass
