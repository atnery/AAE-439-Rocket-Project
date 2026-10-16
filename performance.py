import numpy as np
from scipy.optimize import brentq

def calc_cf(Pc, Pa, At, Ae, gamma):
    if Pc <= 0.0:
        return 0.0
    
    y = gamma
    eps = Ae / At

    # Exit Mach number
    area_mach = lambda M: (1 / M) * ((2 / (y + 1)) * (1 + 0.5 * (y - 1) * M**2)) ** (
        (y + 1) / (2 * (y - 1))
    ) - eps
    Me = 1.0 if np.isclose(eps, 1.0) else brentq(area_mach, 1.0 + 1e-9, 100.0)

    # Exit pressure
    Pe_Pc = (1 + 0.5 * (y - 1) * Me**2) ** (-y / (y - 1))
    Pe = Pe_Pc * Pc

    # Thrust coefficient
    cf = np.sqrt(
        (2 * y**2 / (y - 1))
        * (2 / (y + 1)) ** ((y + 1) / (y - 1))
        * (1 - Pe_Pc ** ((y - 1) / y))
    ) + (Pe - Pa) / Pc * eps

    return max(cf, 0.0)