
from .slicemlp import sliceMLP, sliceMlp_v2,sliceMlp_mulitnet
from .standardmlp import standardMLP
from .adjoint import adjoint,adjoint_patch
from .model_wrapper import model_wrapper


def get_model(n_projections: int, type: str, **kwargs):
    """
    Returns the model with the given name
    Note: For unet the n_projections is not used
    """
    print(type)
    if type == "slicemlp":
        return sliceMLP(n_projections = n_projections,**kwargs)
    elif type == "slicemlp_v2":
        return sliceMlp_v2(n_projections = n_projections,**kwargs)
    elif type == "sliceMlp_mulitnet":
        return sliceMlp_mulitnet(n_projections = n_projections,**kwargs)
    elif type == "adjoint":
        return adjoint()
    elif type == "adjoint_patch":
        return adjoint_patch(**kwargs)
    else:
        raise NotImplementedError(f"Model {type} not implemented")
    