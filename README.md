# Benchmark result

After upgrading from Julia 1.5.4 to 1.6.1:
- Vanilla LBFGS using finite difference: 6.744 s -> 3.188 s
- Single function evaluation: 27.658 μs -> 13.088 μs
- LBFGS using finite difference, with same params as scipy.optimize except that it uses HagerZhang line search: 700.513 ms -> 367.378 ms
- LBFGS using `autodiff = :forward`, with same params as scipy.optimize except that it uses HagerZhang line search: 485.330 ms -> 309.725 ms
- @PharmCat's Newton method with sigmoid: 144.142 ms (on 1.6.1)
- SPGBox: 868.441 μs (on 1.6.1)
- scipy.optimize using PyCall: 14.093 ms -> 10.520 ms (on 1.6.1)

scipy.optimize: 0.448 ± 0.013 s

Single function evaluation:
- Julia: 13.088 μs
- C++: 75 μs
- Python: 1360 μs
