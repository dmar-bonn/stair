class Mapper:
    def __init__(self, args):
        self.class_num = args.class_num
        self.near_far = args.near_far
        self.aabb = args.aabb

    def update_map(self):
        raise NotImplementedError("update_map method is not implemented")

    def render_view(self):
        raise NotImplementedError("render_view method is not implemented")
