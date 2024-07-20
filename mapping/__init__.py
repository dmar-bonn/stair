from .mapper.implicit import ImplicitMapper
from .mapper.explicit import ExplicitMapper


def get_mapper(args, device):
    mapper_type = args.mapper_type

    if mapper_type == "implicit":
        return ImplicitMapper(args, device)
    elif mapper_type == "explicit":
        return ExplicitMapper(args, device)
    else:
        RuntimeError("mapper type not defined")
