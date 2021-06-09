import time

from scipy.optimize import minimize
from scipy.optimize import Bounds

from julia import Main
Main.include("cst_pyjulia.jl")

def do_optimize(fn, xs0, DeltaT, print_elapsed=True):
    tic = time.time()
    method = "L-BFGS-B"
    bounds = Bounds([0.0 for i in range(DeltaT)], [1.0 for i in range(DeltaT)])
    result = minimize(fn, xs0, bounds=bounds, method=method)
    if print_elapsed:
        print("elapsed:", time.time() - tic)
    return result

firm = Main.allocate_firm()
fn = Main.generate_objective_fn(Main.c_greens_all, Main.c_browns_all, firm)
for i in range(20):
    do_optimize(Main.fn, Main.xs0, Main.DeltaT)
    #Main.run()
