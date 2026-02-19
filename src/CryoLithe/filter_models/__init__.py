from .vector import VectorModel, VectorModel_symmetric,VectorModel_real
from .polynomial import PolynomialModel
from .fir import FIRModel
from .cnn import CNN



def get_filter_model(type: str, **kwargs):
    """
    Returns the model with the given name
    Note: For unet the n_projections is not used
    """
    print(type)
    if type == 'vector':
        return VectorModel(**kwargs)
    if type == 'vector_symmetric':
        return VectorModel_symmetric(**kwargs)
    if type == 'VectorModel_real':
        return VectorModel_real(**kwargs)
    if type == 'polynomial':
        return PolynomialModel(**kwargs)
    if type == 'fir':
        return FIRModel(**kwargs)
    if type == 'CNN':
        return CNN(**kwargs)
    else:
        raise NotImplementedError(f"Model {type} not implemented")