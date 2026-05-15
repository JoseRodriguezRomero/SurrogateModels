import OrdinaryDiffEq as ODE
using Plots

node_masses = Dict(
    1 => 1.0,
    2 => 1.0,
    3 => 1.0,
    4 => 1.0,
    5 => 1.0,
    6 => 1.0
)

node_damping_coefficients = Dict(
    1 => 1.0,
    2 => 1.0,
    3 => 1.0,
    4 => 1.0,
    5 => 1.0,
    6 => 1.0
)

node_equilibrium = Dict(
    1 => 2.0,
    2 => 2.0,
    3 => 1.0,
    4 => 6.0,
    5 => 3.0,
    6 => 2.0
)

node_self_couplings = Dict(
    1 => 1.0,
    2 => 1.0,
    3 => 1.0,
    4 => 1.0,
    5 => 1.0,
    6 => 1.0
)

node_couplings = Dict(
    (1,2) => -0.5,
    (1,3) => -0.5,
    (1,4) => -0.5,
    (2,3) => -0.5,
    (2,5) => -0.5,
    (2,6) => -0.5,
    (3,2) => -0.5,
    (3,6) => -0.5,
    (4,6) => -0.5,
    (5,6) => -0.5,
    (6,4) => -0.5,
    (6,5) => -0.5
)

struct NetworkParameters
    node_masses::Dict{Int,Number}
    node_damping_coefficients::Dict{Int,Number}
    node_equilibrium::Dict{Int,Number}
    node_self_couplings::Dict{Int,Number}
    node_couplings::Dict{Tuple{Int,Int},Number}
end

function NetworkParameters(node_masses,node_damping_coefficients,
    node_self_couplings,node_couplings)
    _node_masses = Dict{Int,Number}()
    _node_damping_coefficients = Dict{Int,Number}()
    _node_equilibrium = Dict{Int,Number}()
    _node_self_couplings = Dict{Int,Number}()
    _node_couplings = Dict{Tuple{Int,Int},Number}()

    function copy_data!(source,destination,type)
        for key in keys(source)
            destination[key] = type.(source[key])
        end
    end

    copy_data!(node_masses,_node_masses,Number)
    copy_data!(node_damping_coefficients,_node_damping_coefficients,Number)
    copy_data!(node_equilibrium,_node_equilibrium,Number)
    copy_data!(node_self_couplings,_node_self_couplings,Number)
    copy_data!(node_couplings,_node_couplings,Number)

    return NetworkParameters(_node_masses,_node_damping_coefficients,
        _node_equilibrium,_node_self_couplings,_node_couplings)
end

n_nodes = length(node_masses)

parameters = NetworkParameters(node_masses,node_damping_coefficients,
    node_self_couplings,node_couplings)

coeff_type = typeof(node_masses[1])

X₀ = zeros(coeff_type,n_nodes)
k = zeros(coeff_type,n_nodes,n_nodes)
c = zeros(coeff_type,n_nodes,n_nodes)
for i in keys(node_masses)
    k[i,i] = node_self_couplings[i]
    X₀[i] = node_equilibrium[i]

    c[i,i] = node_damping_coefficients[i]
end

for edge in keys(node_couplings)
    i = edge[1]
    j = edge[2]

    k[j,i] = node_couplings[(i,j)]
end

X_t₀ = X₀ + (ones(Float64,n_nodes) .- 0.5) .* 2.0
dX_t₀ = zeros(Float64,n_nodes)

t₀ = 0.0
t₁ = 10.0
tspan = (t₀,t₁)

function network_problem!(ddu,du,u,parameters,t)
    # node_masses = parameters.node_masses
    # node_damping_coefficients = parameters.node_damping_coefficients
    # node_equilibrium = parameters.node_equilibrium
    # node_self_couplings = parameters.node_self_couplings
    # node_couplings = parameters.node_couplings

    ddu .= - k * (u - X₀) - c * du
end

prob = ODE.SecondOrderODEProblem(network_problem!, dX_t₀, X_t₀, tspan, 
    parameters)
sol = ODE.solve(prob, ODE.DPRKN6(), reltol = 1e-9, abstol = 1e-9)

t = collect(tspan[1]:0.01:tspan[2])

h = 1.0E-3
X = sol.(t)
dX = sol.(t .+ h)

n_nodes = length(node_masses)
X_all = zeros(Float64,length(t),n_nodes)
X_d_all = zeros(Float64,length(t),n_nodes)
X_dd_all = zeros(Float64,length(t),n_nodes)

X_sum = zeros(Float64,length(t))
X_d_sum = zeros(Float64,length(t))
X_dd_sum = zeros(Float64,length(t))
for i in eachindex(X)
    for j in 1:n_nodes
        X_all[i,j] = X[i][n_nodes+j]
        X_d_all[i,j] = X[i][j]
        X_dd_all[i,j] = (X[i][j] - dX[i][j]) / h

        X_sum[i] += X[i][n_nodes+j]
        X_d_sum[i] += X[i][j]
        X_dd_sum[i] += (X[i][j] - dX[i][j]) / h
    end
end

plot(t,X_all)
# plot!(ylims=[0,7])
# plot!(xlims=[0,30])
plot!(frame=:box)
plot!(legend=:outertopright)

# plot(t,X_sum)
# plot!(t,X_d_sum)
# plot(t,X_dd_sum)
