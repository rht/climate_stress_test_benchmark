#include <iostream>
#include <algorithm>
#include <math.h>
#include <vector>
#include <numeric> //std::iota
#include <random>
#include <functional>
#include <chrono>  // for timing the code

std::random_device rd{};
std::mt19937 gen{rd()};

const int MCPATHS = 10;

// depreciation rate
// We couldn't find the values used in Rupert's paper.
const double dg = 0.075;
const double db = 0.15;  // dg < db
// TODO
// @assert (dg[] <= 1) "dg cannot exceed 1"
// @assert (db[] <= 1) "db cannot exceed 1"

const double initial_x = 0.5;

// Cost dynamics parameters for green energy
const double rho_cg = 0.19;  // Rupert paper p5
// Solar parameter in Rupert appendix p58.
const double omega_hat = 0.303;
const double sigma_omega = 0.047;
const double sigma_eta = 0.093;
// Initial cost for green energy
const double cg_initial = 70.0 / 3.6;  // $/GJ, original unit is 70 $/MWh
// Cost of operating existing green energy
// 0.44393708777270424
const double alpha_g = 14.0 / 8760.0 / 0.0036;  // $/GJ
// Rupert appendix eq 32 p38
const double sigma_u = sigma_eta / sqrt(1 + pow(rho_cg, 2.0));

// Cost dynamics parameters for brown energy
// psi is $ of carbon tax per tons of CO2
const double psi = 24.39;  // $/tons

// coal
const double kappa = 0.035;
const double phi_cb = 0.95;
const double sigma_cb = 0.090;
const double cb_initial = 2.18;  // $/GJ
const double chi = 0.1024;  // tons/GJ
const double alpha_b = 1.61;  // $/GJ

const double beta = 1.0;

const double mu = 0.5;

// The weight of variance of r in the utlity formula
const double lambda_utility = 0.1;

// Discount rate
// We use the value from Giglio, Stefano, Matteo Maggiori, and Johannes
// Stroebel. "Very long-run discount rates." The Quarterly Journal of Economics
// 130, no. 1 (2015): 1-53.
// https://academic.oup.com/qje/article/130/1/1/2337985
// Alternatively, we may use the number taken from the plots of page 17 of
// Rupert short paper, which has 3 versions: 0.01, 0.05, 0.09.
const double rho = 0.026;

const int Tstart = 2020;
const int DeltaT = 30;
using vect_int = std::vector<int>;
const auto Ts = []() {
    vect_int _Ts(DeltaT);
    std::iota(std::begin(_Ts), std::end(_Ts), Tstart);
    return _Ts;
}();

const auto full_Ts = []() {
    vect_int Tstart_vec{Tstart};
    vect_int _full_Ts;
    _full_Ts.reserve(Tstart_vec.size() + Ts.size());
    _full_Ts.insert(_full_Ts.end(), Tstart_vec.begin(), Tstart_vec.end());
    _full_Ts.insert(_full_Ts.end(), Ts.begin(), Ts.end());
    return _full_Ts;
}();

const int full_length = full_Ts.size();
const double Energy_total_initial = 420.252047319395 * 1e9;  // GJ
const double brown_energy_fraction = 0.75;

using vect_double = std::vector<double>;
using vect_vect_double = std::vector<std::vector<double>>;

//vect_double do_optimize(std::function<double(vect_double)> fn, vect_double xs0, int DeltaT) {
//    // tic = time()
//    lower = [0.0 for i in 1:DeltaT];
//    upper = [1.0 for i in 1:DeltaT];
//
//    inner_optimizer = Optim.LBFGS;  // 5 min
//    // inner_optimizer = Optim.ConjugateGradient  // 2 min 47 s
//    // inner_optimizer = Optim.GradientDescent  // 2 min
//    println(Symbol(inner_optimizer));
//    result = Optim.optimize(fn, lower, upper, xs0, Optim.Fminbox(inner_optimizer(linesearch = LineSearches.HagerZhang(linesearchmax = 20))), Optim.Options(g_tol = 1e-5, f_tol = 2.2e-9));
//    //result = Optim.optimize(fn, lower, upper, xs0, Optim.Fminbox(inner_optimizer()), autodiff = :forward)
//    //println("elapsed: ", time() - tic)
//    return result;
//}

double sum(vect_double vect) {
    double sum_of_elems = 0.0;
    for (auto& n : vect)
        sum_of_elems += n;
    return sum_of_elems;
}

double calculate_cost_g(double cg, double x, double delta_E, double Eg, double alpha_g) {
    return cg * x * delta_E + alpha_g * pow(Eg, beta);
}

double calculate_cost_b(double cb, double tax, double x, double delta_E, double Eb, double alpha_b) {
    double delta_Eb = (1 - x) * delta_E;
    double tax_term = tax * (delta_Eb + Eb);
    return cb * delta_Eb + alpha_b * pow(Eb, beta) + tax_term;
}

vect_vect_double evolve_cg(double omega_hat, double sigma_omega, double sigma_u, double cg_initial) {
    // Rupert appendix p38
    // We generate the cost evolution for every monte carlo
    // path, and then we append them into the list
    // c_greens_all.
    vect_vect_double c_greens_all;
    std::normal_distribution<> dist{omega_hat, sigma_omega};
    std::normal_distribution<> dist_ut{0, sigma_u};
    for (int n = 1; n <= MCPATHS; n++) {
        double omega_cg = dist(gen);
        vect_double c_greens{cg_initial};
        vect_double ut_greens;
        for (unsigned int j = 0; j < Ts.size(); j++) {
            double ut = dist_ut(gen);
            ut_greens.push_back(ut);
            double cg = c_greens.back();
            // Wright's law
            double ut_minus1;
            if (j == 0) {
                ut_minus1 = 0.0;
            } else {
                ut_minus1 = ut_greens[j - 1];
            }
            double cg_next = cg * exp(-omega_cg + ut + rho_cg * ut_minus1);
            c_greens.push_back(cg_next);
        }
        c_greens_all.push_back(c_greens);
    }
    return c_greens_all;
}

vect_vect_double evolve_cb(double sigma_cb, double cb_initial, double kappa, double phi_cb) {
    vect_vect_double c_browns_all;
    std::normal_distribution<> dist{0, sigma_cb};
    for (int n = 1; n <= MCPATHS; n++) {
        vect_double c_browns{cb_initial};
        for (unsigned int j = 0; j < Ts.size(); j++) {
            double cb = c_browns.back();
            // AR(1)
            // Equation 25 of Rupert appendix
            double m_cb = kappa / (1.0 - phi_cb);
            double epsilon_cb = dist(gen);
            double cb_next = cb * exp((1.0 - phi_cb) * (m_cb - log(cb)) + epsilon_cb);
            c_browns.push_back(cb_next);
        }
        c_browns_all.push_back(c_browns);
    }
    return c_browns_all;
}

vect_double calculate_taxes(double tax_initial) {
    double tax = tax_initial;
    vect_double taxes{tax};
    for (unsigned int j = 0; j < Ts.size(); j++) {
        tax += 10.0 * chi;
        taxes.push_back(tax);
    }
    return taxes;
}


class Firm {
    public:
        double Energy_total;
        double Eg;
        vect_double E_greens = vect_double(full_length);
        double Eb;
        vect_double E_browns = vect_double(full_length);
        double delta_E;
        //vect_double denominators;
        vect_double denominators = vect_double(full_length);
        double price;
        //vect_double numerators;
        vect_double numerators = vect_double(full_length);
        double alpha_g;
        double alpha_b;

        Firm() {
            Energy_total = Energy_total_initial;
            // Rupert appendix p29
            // Note, the actual value could be taken from Rupert appendix p17,
            // table 5.
            // Initialize first element of all the time series at t = 2020
            // GJ/yr, useful energy at t0
            Eg = (1 - brown_energy_fraction) * Energy_total;
            // Time series of green energy
            E_greens[0] = Eg;
            // Rupert appendix p29
            // GJ/yr, useful energy at t0
            Eb = brown_energy_fraction * Energy_total;
            // Time series of brown energy
            E_browns[0] = Eb;
            // Total depreciation of energy
            delta_E = dg * Eg + db * Eb;

            price = 0.0;
            // alpha_g and alpha_b are set as attributes of a Firm because they may
            // vary across firms in the multifirm situation.
            alpha_g = ::alpha_g;
            alpha_b = ::alpha_b;
        }

        double calculate_production_cost(double cg, double cb, double x, double tax) {
            double production_cost = (
                    calculate_cost_g(cg, x, delta_E, Eg, alpha_g) +
                    calculate_cost_b(cb, tax, x, delta_E, Eb, alpha_b)
                    );
            return production_cost;
        }

        void step1(double x, int j) {
            // Update Eg, Eb, and delta_E
            // TODO
            //@assert isapprox(firm.Energy_total, firm.Eg + firm.Eb)
            // Doyne equation 18
            Eg = Eg * (1 - dg) + x * delta_E;
            // Doyne equation 19
            Eb = Eb * (1 - db) + (1 - x) * delta_E;
            // Update delta_E
            // Note: Eg and Eb update use the value of delta_E
            // before this update. The order of update is important!
            delta_E = dg * Eg + db * Eb;

            E_greens[j] = Eg;
            E_browns[j] = Eb;
        }

        void step2(int t, double x, double tax, double cg, double cb, double production_cost, int j) {
            // For calculating numerator and denominator
            // Discounted profit associated with year t + tau
            // t - Tstart is tau
            double supply = Eg + Eb;
            double discount = exp(-rho * (t - Tstart));
            double numerator = discount * (
                    price * supply - production_cost
                    );
            // numerator = max(numerator, 0)
            //numerators.push_back(numerator);
            numerators[j] = numerator;
            double denominator = discount * (cg * x + (cb + tax) * (1 - x)) * delta_E;
            //denominators.push_back(denominator);
            denominators[j] = denominator;
        }
};

std::function<double(vect_double)> generate_objective_fn(vect_vect_double c_greens_all, vect_vect_double c_browns_all) {
    // c_greens_all is a list of time series list of cost of
    // green energy and likewise c_browns_all is brown energy.
    // They have the length of MCPATHS
    vect_double taxes = calculate_taxes(0.0);
    // By default, `measures` is a list of returns of the company.
    vect_double measures(MCPATHS);
    auto _obj_fn = [c_greens_all, c_browns_all, taxes, measures](vect_double xs) mutable {
        // TODO
        //@assert length(xs) == length(Ts)
        for (int n = 0; n < MCPATHS; n++) {
            vect_double c_greens = c_greens_all[n];
            vect_double c_browns = c_browns_all[n];

            int t = Tstart;
            double x = initial_x;
            double tax = taxes[0];
            double cg = c_greens[0];
            double cb = c_browns[0];
            Firm firm = Firm();
            double production_cost = firm.calculate_production_cost(cg, cb, x, tax);
            firm.price = (1.0 + mu) * production_cost / firm.Energy_total;
            firm.step2(t, x, tax, cg, cb, production_cost, 1);

            for (unsigned int j = 0; j < Ts.size(); j++) {
                // Order of update is important. step1() must
                // happen first, because it must use the value
                // of x before the update. And the subsequent
                // steps require the updated Eg, Eb, and
                // delta_E
                double t = Ts[j];
                firm.step1(x, j + 1);
                double x = xs[j];
                double tax = taxes[j + 1];
                double cg = c_greens[j + 1];
                double cb = c_browns[j + 1];
                double production_cost = firm.calculate_production_cost(cg, cb, x, tax);
                firm.step2(t, x, tax, cg, cb, production_cost, j + 1);
            }

            double measure = sum(firm.numerators) / sum(firm.denominators);
            measures[n] = measure;
        }

        double sum_measures = sum(measures);
        double mean_measures = sum_measures / measures.size();
        double sq_sum = std::inner_product(measures.begin(), measures.end(), measures.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / measures.size() - mean_measures * mean_measures);
        double U = mean_measures - lambda_utility * stdev;
        // Reverse the sign because we only have `minimize` function
        double out = -U;

        return out;
    };
    return _obj_fn;
}

const auto c_greens_all = evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial);
const auto c_browns_all = evolve_cb(sigma_cb, cb_initial, kappa, phi_cb);

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    vect_double xs0;
    for (int i = 0; i < DeltaT; i++) {
        xs0.push_back(std::min(1.0, initial_x));
    }

    auto fn = generate_objective_fn(c_greens_all, c_browns_all);
    std::cout << "fn execution\n";
    auto t1 = high_resolution_clock::now();
    double out = fn(xs0);
    auto t2 = high_resolution_clock::now();
    std::cout << out << "\n";
    auto mus_int = duration_cast<microseconds>(t2 - t1);
    std::cout << "elapsed time\n";
    std::cout << mus_int.count() << "\n";
    //do_optimize(fn, xs0, DeltaT);
    return 0;
}
