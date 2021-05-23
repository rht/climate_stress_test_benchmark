import math
import os
import time

from typing import List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds


# For deterministic result
rng = np.random.default_rng(1337)

MCPATHS = 10

# depreciation rate
# We couldn't find the values used in Rupert's paper.
dg = 0.075
db = 0.15  # dg < db
assert dg <= 1, "dg cannot exceed 1"
assert db <= 1, "db cannot exceed 1"

initial_x = 0.5

# Cost dynamics parameters for green energy
rho_cg = 0.19  # Rupert paper p5
# Solar parameter in Rupert appendix p58.
omega_hat = 0.303
sigma_omega = 0.047
sigma_eta = 0.093
# Initial cost for green energy
cg_initial = 70 / 3.6  # $/GJ, original unit is 70 $/MWh
# Cost of operating existing green energy
# 0.44393708777270424
alpha_g = 14 / 8760 / 0.0036  # $/GJ
# Rupert appendix eq 32 p38
sigma_u = sigma_eta / np.sqrt(1 + rho_cg ** 2)

# Cost dynamics parameters for brown energy
# psi is $ of carbon tax per tons of CO2
psi = 24.39  # $/tons

# coal
kappa = 0.035
phi_cb = 0.95
sigma_cb = 0.090
cb_initial = 2.18  # $/GJ
chi = 0.1024  # tons/GJ
alpha_b = 1.61  # $/GJ

beta = 1

mu = 0.5

# The weight of variance of r in the utlity formula
lambda_utility = 0.1

# Discount rate
# We use the value from Giglio, Stefano, Matteo Maggiori, and Johannes
# Stroebel. "Very long-run discount rates." The Quarterly Journal of Economics
# 130, no. 1 (2015): 1-53.
# https://academic.oup.com/qje/article/130/1/1/2337985
# Alternatively, we may use the number taken from the plots of page 17 of
# Rupert short paper, which has 3 versions: 0.01, 0.05, 0.09.
rho = 0.026

Tstart = 2020
DeltaT = 30
Ts = list(range(2021, 2021 + DeltaT))
full_Ts = [2020] + Ts
Energy_total_initial = 420.252047319395 * 1e9  # GJ
brown_energy_fraction = 0.75

def do_optimize(fn, xs0, DeltaT, print_elapsed=True):
    tic = time.time()
    method = "L-BFGS-B"
    bounds = Bounds([0.0 for i in range(DeltaT)], [1.0 for i in range(DeltaT)])
    result = minimize(fn, xs0, bounds=bounds, method=method)
    if print_elapsed:
        print("elapsed:", time.time() - tic)
    return result


def calculate_cost_g(
    cg: float, x: float, delta_E: float, Eg: float, alpha_g: float
) -> float:
    return cg * x * delta_E + alpha_g * (Eg ** beta)


def calculate_cost_b(
    cb: float, tax: float, x: float, delta_E: float, Eb: float, alpha_b: float
) -> float:
    delta_Eb = (1 - x) * delta_E
    tax_term = tax * (delta_Eb + Eb)
    return (cb * delta_Eb + alpha_b * (Eb ** beta)) + tax_term


def evolve_cg(
    omega_hat: float, sigma_omega: float, sigma_u: float, cg_initial: float
) -> List[List[float]]:
    # Rupert appendix p38
    # We generate the cost evolution for every monte carlo
    # path, and then we append them into the list
    # c_greens_all.
    c_greens_all = []
    for n in range(MCPATHS):
        omega_cg = rng.normal(omega_hat, sigma_omega)
        ut_greens = rng.normal(0, sigma_u, len(Ts))
        c_greens = [cg_initial]
        for j in range(len(Ts)):
            ut = ut_greens[j]
            cg = c_greens[-1]
            # Wright's law
            if j == 0:
                ut_minus1 = 0
            else:
                ut_minus1 = ut_greens[j - 1]
            cg_next = cg * math.exp(-omega_cg + ut + rho_cg * ut_minus1)
            c_greens.append(cg_next)
        c_greens_all.append(c_greens)
    return c_greens_all


def evolve_cb(
    sigma_cb: float, cb_initial: float, kappa: float, phi_cb: float
) -> List[List[float]]:
    c_browns_all = []
    for n in range(MCPATHS):
        epsilon_cb = rng.normal(0, sigma_cb, len(Ts))
        c_browns = [cb_initial]
        for j in range(len(Ts)):
            cb = c_browns[-1]
            # AR(1)
            # Equation 25 of Rupert appendix
            m_cb = kappa / (1 - phi_cb)
            cb_next = cb * math.exp(
                (1 - phi_cb) * (m_cb - math.log(cb)) + epsilon_cb[j]
            )
            c_browns.append(cb_next)
        c_browns_all.append(c_browns)
    return c_browns_all


def calculate_taxes(tax_initial: float) -> List[float]:
    tax = tax_initial
    taxes = [tax]
    for t in Ts:
        tax += 10.0 * chi
        taxes.append(tax)
    return taxes


class Firm:
    __slots__ = (
        "Energy_total",
        "Eg",
        "E_greens",
        "Eb",
        "E_browns",
        "delta_E",
        "denominators",
        "price",
        "numerators",
        "alpha_g",
        "alpha_b",
    )

    def __init__(
        self,
        alpha_g: float,
        alpha_b: float,
        energy_total: float,
        capacity_multiplier: float = 1.0,
    ) -> None:
        self.Energy_total: float = energy_total
        # Rupert appendix p29
        # Note, the actual value could be taken from Rupert appendix p17,
        # table 5.
        # Initialize first element of all the time series at t = 2020
        self.Eg: float = (1 - brown_energy_fraction) * self.Energy_total
        # Time series of green energy
        self.E_greens: List[float] = [self.Eg]  # GJ/yr, useful energy at t0
        # Rupert appendix p29
        self.Eb: float = brown_energy_fraction * self.Energy_total
        # Time series of brown energy
        self.E_browns: List[float] = [self.Eb]  # GJ/yr, useful energy at t0
        # Total depreciation of energy
        self.delta_E: float = dg * self.Eg + db * self.Eb

        self.numerators: List[float] = []
        self.denominators: List[float] = []
        self.price: Optional[float] = None
        # alpha_g and alpha_b are set as attributes of a Firm because they may
        # vary across firms in the multifirm situation.
        self.alpha_g: float = alpha_g
        self.alpha_b: float = alpha_b

    def calculate_production_cost(
        self, cg: float, cb: float, x: float, tax: float
    ) -> float:
        production_cost = calculate_cost_g(
            cg, x, self.delta_E, self.Eg, self.alpha_g
        ) + calculate_cost_b(cb, tax, x, self.delta_E, self.Eb, self.alpha_b)
        return production_cost

    def step1(self, t: int, x: float) -> None:
        # Update Eg, Eb, and delta_E
        # Doyne equation 18
        self.Eg = self.Eg * (1 - dg) + x * self.delta_E
        # Doyne equation 19
        self.Eb = self.Eb * (1 - db) + (1 - x) * self.delta_E
        assert abs(self.Energy_total - (self.Eg + self.Eb)) / self.Energy_total < 1e-9
        # Update delta_E
        # Note: Eg and Eb update use the value of delta_E
        # before this update. The order of update is important!
        self.delta_E = dg * self.Eg + db * self.Eb

        self.E_greens.append(self.Eg)
        self.E_browns.append(self.Eb)

    def step2(
        self, t: int, x: float, tax: float, cg: float, cb: float, production_cost: float
    ) -> None:
        # For calculating numerator and denominator
        # Discounted profit associated with year t + tau
        # t - Tstart is tau
        supply = self.Eg + self.Eb
        discount = math.exp(-rho * (t - Tstart))
        numerator = discount * (self.price * supply - production_cost)
        # Uncomment this to truncate profit to be >= 0
        # numerator = max(numerator, 0)
        self.numerators.append(numerator)

        denominator = discount * (cg * x + (cb + tax) * (1 - x)) * self.delta_E
        self.denominators.append(denominator)


def generate_objective_fn(
    c_greens_all: List[List[float]],
    c_browns_all: List[List[float]],
):
    # c_greens_all is a list of time series list of cost of
    # green energy and likewise c_browns_all is brown energy.
    # They have the length of MCPATHS
    taxes = calculate_taxes(0.0)

    def _obj_fn(xs):
        # By default, `measures` is a list of returns of the company.
        measures = []
        assert len(xs) == len(Ts)
        for n in range(MCPATHS):
            c_greens = c_greens_all[n]
            c_browns = c_browns_all[n]

            t = Tstart
            x = initial_x
            tax = taxes[0]
            cg = c_greens[0]
            cb = c_browns[0]
            firm = Firm(alpha_g, alpha_b, Energy_total_initial)
            production_cost = firm.calculate_production_cost(cg, cb, x, tax)
            firm.price = (1 + mu) * production_cost / firm.Energy_total
            firm.step2(t, x, tax, cg, cb, production_cost)

            for j, t in enumerate(Ts):
                # Order of update is important. step1() must
                # happen first, because it must use the value
                # of x before the update. And the subsequent
                # steps require the updated Eg, Eb, and
                # delta_E
                firm.step1(t, x)
                x = xs[j]
                tax = taxes[j + 1]
                cg = c_greens[j + 1]
                cb = c_browns[j + 1]
                production_cost = firm.calculate_production_cost(cg, cb, x, tax)
                firm.step2(t, x, tax, cg, cb, production_cost)

            measure = sum(firm.numerators) / sum(firm.denominators)
            measures.append(measure)

        U = np.mean(measures) - lambda_utility * np.var(measures)
        # Reverse the sign because we only have `minimize` function
        out = -U

        return out

    return _obj_fn


c_greens_all = evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial)
c_browns_all = evolve_cb(sigma_cb, cb_initial, kappa, phi_cb)

xs0 = [min(1, initial_x) for i in range(DeltaT)]

if __name__ == "__main__":
    fn = generate_objective_fn(c_greens_all, c_browns_all)
    result = do_optimize(fn, xs0, DeltaT)
