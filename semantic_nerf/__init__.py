from .semantic_densevoxel import SemanticDenseVoxel
from dotmap import DotMap


def get_model(args, device):
    model_type = args.model_type
    if model_type == "densevoxel":
        return SemanticDenseVoxel(args, device)
    else:
        raise NotImplementedError


def load_model(model_file, device):
    args = DotMap(model_file["kwargs"])
    model_type = args.model_type
    if model_type == "densevoxel":
        model = SemanticDenseVoxel(args, device)
        model.load_model(model_file)
        return model
    else:
        raise NotImplementedError
