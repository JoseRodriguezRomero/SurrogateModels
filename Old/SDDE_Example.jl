using StochasticDelayDiffEq
using Plots, ForwardDiff

# 1. Define the drift function (deterministic part)
# f(u, h, p, t)
# h is the history function: h(p, t-tau)
function f(u, h, p, t)
    # tau = 0.2
    # We use h(p, t-tau) to get the value of u at the delayed time
    # delayed_u = h(p, t - tau)
    # return 0.8 * u * (1.0 - delayed_u)

    du = similar(u)

    du[1] = -sign(u[1])
    du[2] = -ForwardDiff.derivative(tt -> h(p, tt - 0.5), t)[1]
    du[3] = ForwardDiff.derivative(tt -> h(p, tt - 2.5), t)[2]

    return du
end

# 2. Define the diffusion function (stochastic part)
function g(u, h, p, t)
    return 0.0 * u
end

# 3. Define the history function
# This returns the value of the system for all t <= 0
h(p, t) = [0.0, 0.0, 0.0]

# 4. Problem Setup
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 5.0)
lags = [0.2]

# 5. Define the SDDE Problem
prob = SDDEProblem(f, g, u0, h, tspan; constant_lags = lags)

# 6. Solve the equation
# RKMil() is the Rossler-Komori Strong Order 1.0 L-stable method for SDDEs
sol = solve(prob, EM(), dt=0.001)

# 7. Visualize
plot(sol)
