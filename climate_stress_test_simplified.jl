using Profile
using StatProfilerHTML
using BenchmarkTools

import Random
import Statistics

import Optim
import LineSearches


# For deterministic result
Random.seed!(1337)

const MCPATHS = 10

# depreciation rate
# We couldn't find the values used in Rupert's paper.
const dg = Ref(0.075)
const db = Ref(0.15)  # dg < db
@assert (dg[] <= 1) "dg cannot exceed 1"
@assert (db[] <= 1) "db cannot exceed 1"

const initial_x = 0.5

# Cost dynamics parameters for green energy
const rho_cg = 0.19  # Rupert paper p5
# Solar parameter in Rupert appendix p58.
const omega_hat = 0.303
const sigma_omega = 0.047
const sigma_eta = 0.093
# Initial cost for green energy
const cg_initial = 70 / 3.6  # $/GJ, original unit is 70 $/MWh
# Cost of operating existing green energy
# 0.44393708777270424
const alpha_g = Ref(14 / 8760 / 0.0036)  # $/GJ
# Rupert appendix eq 32 p38
const sigma_u = sigma_eta / sqrt(1 + rho_cg ^ 2)

# Cost dynamics parameters for brown energy
# psi is $ of carbon tax per tons of CO2
const psi = 24.39  # $/tons

# coal
const kappa = 0.035
const phi_cb = 0.95
const sigma_cb = 0.090
const cb_initial = 2.18  # $/GJ
const chi = 0.1024  # tons/GJ
const alpha_b = Ref(1.61)  # $/GJ

const beta = Ref(1.0)

const mu = 0.5

# The weight of variance of r in the utlity formula
const lambda_utility = Ref(0.1)

# Discount rate
# We use the value from Giglio, Stefano, Matteo Maggiori, and Johannes
# Stroebel. "Very long-run discount rates." The Quarterly Journal of Economics
# 130, no. 1 (2015): 1-53.
# https://academic.oup.com/qje/article/130/1/1/2337985
# Alternatively, we may use the number taken from the plots of page 17 of
# Rupert short paper, which has 3 versions: 0.01, 0.05, 0.09.
const rho = 0.026

const Tstart = 2020
const DeltaT = 30
const Ts = collect(2021:(2020 + DeltaT))
const full_Ts = [2020; Ts]
const full_length = length(full_Ts)
const Energy_total_initial = 420.252047319395 * 1e9  # GJ
const brown_energy_fraction = Ref(0.75)

function do_optimize(fn, xs0, DeltaT)
    tic = time()
    lower = [0.0 for i in 1:DeltaT]
    upper = [1.0 for i in 1:DeltaT]

    inner_optimizer = Optim.LBFGS  # 5 min
    # inner_optimizer = Optim.ConjugateGradient  # 2 min 47 s
    # inner_optimizer = Optim.GradientDescent  # 2 min
    println(Symbol(inner_optimizer))

    linesearch = LineSearches.HagerZhang(linesearchmax = 20)
    #linesearch = LineSearches.BackTracking(iterations = 20)
    #linesearch = LineSearches.MoreThuente(maxfev = 20)

    # baseline
    #result = Optim.optimize(fn, lower, upper, xs0, Optim.Fminbox(inner_optimizer()))

    # With same params as scipy.optimize except for line search algorithm
    options = Optim.Options(g_tol = 1e-5, f_tol = 2.2e-9)
    result = Optim.optimize(fn, lower, upper, xs0, Optim.Fminbox(inner_optimizer(linesearch = linesearch)), options)
    println("elapsed: ", time() - tic)
    return result
end

function calculate_cost_g(cg, x, delta_E, Eg, alpha_g)
    return cg * x * delta_E + alpha_g * (Eg ^ beta[])
end

function calculate_cost_b(cb, tax, x, delta_E, Eb, alpha_b)
    delta_Eb = (1 - x) * delta_E
    tax_term = tax * (delta_Eb + Eb)
    return (
        cb * delta_Eb +
        alpha_b * (Eb ^ beta[]) +
        tax_term
    )
end

function evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial)
    # Rupert appendix p38
    # We generate the cost evolution for every monte carlo
    # path, and then we append them into the list
    # c_greens_all.
    c_greens_all = Array{Float64,1}[]
    for n in 1:MCPATHS
        omega_cg = omega_hat + randn() * sigma_omega
        ut_greens = randn(length(Ts)) * sigma_u
        c_greens = [cg_initial]
        for j in 1:length(Ts)
            ut = ut_greens[j]
            cg = c_greens[end]
            # Wright's law
            if j == 1
                ut_minus1 = 0
            else
                ut_minus1 = ut_greens[j - 1]
            end
            cg_next = cg * exp(-omega_cg + ut + rho_cg * ut_minus1)
            push!(c_greens, cg_next)
        end
        push!(c_greens_all, c_greens)
    end
    return c_greens_all
end

function evolve_cb(sigma_cb, cb_initial, kappa, phi_cb)
    c_browns_all = Array{Float64,1}[]
    for n in 1:MCPATHS
        epsilon_cb = randn(length(Ts)) * sigma_cb
        c_browns = [cb_initial]
        for j in 1:length(Ts)
            cb = c_browns[end]
            # AR(1)
            # Equation 25 of Rupert appendix
            m_cb = kappa / (1 - phi_cb)
            cb_next = cb * exp((1 - phi_cb) * (m_cb - log(cb)) + epsilon_cb[j])
            push!(c_browns, cb_next)
        end
        push!(c_browns_all, c_browns)
    end
    return c_browns_all
end

function calculate_taxes(tax_initial::Float64)
    tax = tax_initial
    taxes = [tax]
    for t in Ts
        tax += 10.0 * chi
        push!(taxes, tax)
    end
    return taxes
end


abstract type AbstractFirm end

mutable struct Firm <: AbstractFirm
    Energy_total::Float64
    Eg::Float64
    E_greens::Array{Float64,1}
    Eb::Float64
    E_browns::Array{Float64,1}
    delta_E::Float64
    denominators::Array{Float64,1}
    price::Float64
    numerators::Array{Float64,1}
    alpha_g::Float64
    alpha_b::Float64
end

function init!(firm::Firm)
    firm.Energy_total = Energy_total_initial
    # Rupert appendix p29
    # Note, the actual value could be taken from Rupert appendix p17,
    # table 5.
    # Initialize first element of all the time series at t = 2020
    # GJ/yr, useful energy at t0
    firm.Eg = (1 - brown_energy_fraction[]) * firm.Energy_total
    # Time series of green energy
    firm.E_greens[1] = firm.Eg
    # Rupert appendix p29
    # GJ/yr, useful energy at t0
    firm.Eb = brown_energy_fraction[] * firm.Energy_total
    # Time series of brown energy
    firm.E_browns[1] = firm.Eb
    # Total depreciation of energy
    firm.delta_E = dg[] * firm.Eg + db[] * firm.Eb

    firm.price = 0.0
    # alpha_g and alpha_b are set as attributes of a Firm because they may
    # vary across firms in the multifirm situation.
    firm.alpha_g = alpha_g[]
    firm.alpha_b = alpha_b[]
end

function allocate_firm()
    return Firm(
                0.0,
                0.0,
                Array{Float64,1}(undef, full_length),
                0.0,
                Array{Float64,1}(undef, full_length),
                0.0,
                Array{Float64,1}(undef, full_length),
                0.0,
                Array{Float64,1}(undef, full_length),
                0.0,
                0.0)
end

function calculate_production_cost(firm::AbstractFirm, cg, cb, x, tax)
    production_cost = (
        calculate_cost_g(cg, x, firm.delta_E, firm.Eg, firm.alpha_g) +
        calculate_cost_b(cb, tax, x, firm.delta_E, firm.Eb, firm.alpha_b)
    )
    return production_cost
end

function step1!(firm::AbstractFirm, x, j)
    # Update Eg, Eb, and delta_E
    @assert isapprox(firm.Energy_total, firm.Eg + firm.Eb)
    # Doyne equation 18
    firm.Eg = firm.Eg * (1 - dg[]) + x * firm.delta_E
    # Doyne equation 19
    firm.Eb = firm.Eb * (1 - db[]) + (1 - x) * firm.delta_E
    # Update delta_E
    # Note: Eg and Eb update use the value of delta_E
    # before this update. The order of update is important!
    firm.delta_E = dg[] * firm.Eg + db[] * firm.Eb

    firm.E_greens[j] = firm.Eg
    firm.E_browns[j] = firm.Eb
end

function step2!(firm::AbstractFirm, t, x, tax, cg, cb, production_cost, j)
    # For calculating numerator and denominator
    # Discounted profit associated with year t + tau
    # t - Tstart is tau
    supply = firm.Eg + firm.Eb
    discount = exp(-rho * (t - Tstart))
    numerator = discount * (
        firm.price * supply - production_cost
    )
    # numerator = max(numerator, 0)
    firm.numerators[j] = numerator
    denominator = discount * (cg * x + (cb + tax) * (1 - x)) * firm.delta_E
    firm.denominators[j] = denominator
end

function generate_objective_fn(c_greens_all, c_browns_all, firm)
    # c_greens_all is a list of time series list of cost of
    # green energy and likewise c_browns_all is brown energy.
    # They have the length of MCPATHS
    taxes = calculate_taxes(0.0)
    # By default, `measures` is a list of returns of the company.
    measures = Array{Float64,1}(undef, MCPATHS)
    function _obj_fn(xs)
        @assert length(xs) == length(Ts)
        for n in 1:MCPATHS
            c_greens = c_greens_all[n]
            c_browns = c_browns_all[n]

            t = Tstart
            x = initial_x
            tax = taxes[1]
            cg = c_greens[1]
            cb = c_browns[1]
            init!(firm)
            production_cost = calculate_production_cost(firm, cg, cb, x, tax)
            firm.price = (1 + mu) * production_cost / firm.Energy_total
            step2!(firm, t, x, tax, cg, cb, production_cost, 1)

            for (j, t) in enumerate(Ts)
                # Order of update is important. step1() must
                # happen first, because it must use the value
                # of x before the update. And the subsequent
                # steps require the updated Eg, Eb, and
                # delta_E
                step1!(firm, x, j + 1)
                x = xs[j]
                tax = taxes[j + 1]
                cg = c_greens[j + 1]
                cb = c_browns[j + 1]
                production_cost = calculate_production_cost(firm, cg, cb, x, tax)
                step2!(firm, t, x, tax, cg, cb, production_cost, j + 1)
            end

            measure = sum(firm.numerators) / sum(firm.denominators)
            measures[n] = measure
        end

        U = Statistics.mean(measures) - lambda_utility[] * Statistics.var(measures)
        # Reverse the sign because we only have `minimize` function
        out = -U

        return out
    end
    return _obj_fn
end

const c_greens_all = evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial)
const c_browns_all = evolve_cb(sigma_cb, cb_initial, kappa, phi_cb)

const xs0 = [min(1, initial_x) for i in 1:DeltaT]

if abspath(PROGRAM_FILE) == @__FILE__
    # The let...end needs to be commented out, otherwise, @btime can't detect
    # `fn`.
    #let
        # We preallocate the firm struct once for all to reduce allocations.
        firm = allocate_firm()
        fn = generate_objective_fn(c_greens_all, c_browns_all, firm)
        #@profilehtml fn(xs0)
        #@btime fn(xs0)
        #Profile.print(format=:flat, sortedby=:count)
        @btime result = do_optimize(fn, xs0, DeltaT)
        #result = do_optimize(fn, xs0, DeltaT)
        #println(result)
        #println(Optim.minimizer(result))
    #end
end
