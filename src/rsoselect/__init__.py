"""rsoselect: ridge selection operator (RSO) for variable selection.

    RSO    linear model              y = X beta + eps
    PLM    partially linear model    y = X beta + f(Z) + eps
    GPLM   generalized PLM           g(E y) = X beta + f(Z)   (local scoring)
    mcd    modified coordinate descent on simplices
"""

from .gplm import GPLM
from .linear import RSO, rso
from .mcd import mcd
from .plm import PLM
from .tuning import aic, bic, default_taus

__all__ = ["RSO", "rso", "PLM", "GPLM", "mcd", "bic", "aic", "default_taus"]
__version__ = "0.2.1"
