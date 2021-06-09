using BenchmarkTools
import LBFGSB
import Optim
import PyCall

const scipy_optimize = PyCall.pyimport("scipy.optimize")

function fn(x)
    y = 0.25 * (x[1] - 1)^2
    for i = 2:length(x)
        y += (x[i] - x[i-1]^2)^2
    end
    4y
end

function do_optimize_scipy(fn, x0)
    method = "L-BFGS-B"
    bounds = scipy_optimize.Bounds([isodd(i) ? 1e0 : -1e2 for i = 1:length(x0)], [1e2 for i = 1:length(x0)])
    result = scipy_optimize.minimize(fn, x0, bounds=bounds, method=method)
    return result
end

promote_objtype(x, f) = Optim.OnceDifferentiable(f, x, real(zero(eltype(x))))

const n = 25  # dimension of the problem
const x0 = fill(3e0, n)
let
    d = promote_objtype(x0, fn)
    function finite_difference_g!(z, x)
        z .= Optim.gradient!(d, x)
    end

    # For LBFGSB.jl
    bounds = zeros(3, n)
    for i = 1:n
        bounds[1,i] = 2  # represents the type of bounds imposed on the variables:
                         #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
        bounds[2,i] = isodd(i) ? 1e0 : -1e2  #  the lower bound on x, of length n.
        bounds[3,i] = 1e2  #  the upper bound on x, of length n.
    end

    println("Benchmarking single function execution")
    @btime fn(x0)
    println("Benchmarking LBFGSB.jl")
    nmax = length(x0)
    mmax = 10
    ftol = 2.2204460492503131e-09
    factr = ftol / Base.eps(Float64)
    lbfgsb_optimizer = LBFGSB.L_BFGS_B(nmax, mmax)
    @btime $lbfgsb_optimizer(fn, $finite_difference_g!, x0, $bounds, m=10, factr=$factr, pgtol=1e-5, iprint=-1, maxfun=15000, maxiter=15000)
    fout, xout = lbfgsb_optimizer(fn, finite_difference_g!, x0, bounds, m=10, factr=factr, pgtol=1e-5, iprint=-1, maxfun=15000, maxiter=15000)
    println(fout)
    println("Benchmarking scipy.optimize(method='L-BFGS-B')")
    @btime do_optimize_scipy(fn, x0)
    result = do_optimize_scipy(fn, x0)
    println(result["fun"])
end
