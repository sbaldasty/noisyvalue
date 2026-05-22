from .util import fresh_name
from sympy.stats import Normal

def gaussian(loc, scale):
    return lambda: Normal(fresh_name(), loc, scale)
