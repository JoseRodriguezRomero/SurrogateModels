using Plots, ForwardDiff, LaTeXStrings, Measures
using DifferentialEquations

function σ(K, c)
    return sqrt(K*c)
end

function B(K,c)
    return sqrt((π^2)/(3*(σ(K, c)^2)))
end

function A(K,c)
    return K / log(1 + exp(K * B(K,c)))
end

function Q(K, φ, c, t)
    return A(K, c) * log(1 + exp(B(K, c) * (K - t / φ)))
end

function dQ(K, φ, c, t)
    return ForwardDiff.derivative(t -> Q(K, φ, c, t), t)
    # return - (A(K, φ, c) * B(K, c) / φ) / (1 + exp(-B(K, c) * (K - t / φ)))
end

function ddQ(K, φ, c, t)
    return ForwardDiff.derivative(t -> dQ(K, φ, c, t), t)
    # aux_exp = exp(-B(K, c) * (K - t / φ))
    # return (A(K, φ, c) * (B(K, c) / φ)^2) * (aux_exp / (1 + aux_exp)^2)
end

function normal(K, φ, c, t)
    c1 = (1/(φ^2))*(1/sqrt(2*π*(σ(K, c)^2)))
    c2 = exp(-((K - t/φ)^2) / (2 * (σ(K, c)^2)))
    return c1 * c2
end

function plot_q(c, φ)
    K1 = 10.0
    K2 = 20.0
    K3 = 30.0
    K4 = 40.0

    t0 = 0.0
    t1 = 60.0
    t = collect(t0:0.01:t1)

    q_values = Q.(K1, φ, c, t)
    p = plot(t, q_values, label = L"K = 10")

    q_values = Q.(K2, φ, c, t)
    plot!(t, q_values, label = L"K = 20")

    q_values = Q.(K3, φ, c, t)
    plot!(t, q_values, label = L"K = 30")

    q_values = Q.(K4, φ, c, t)
    plot!(t, q_values, label = L"K = 40")

    plot!(ylabel=L"$Q_{1,2} (t)$")

    plot!(frame = :box)
    plot!(xlims = [t0,t1])
    plot!(ylims=[-5,50])
    plot!(xticks = (t0:10:t1,[]))
    return p
end

function plot_dq(c, φ)
    K1 = 10.0
    K2 = 20.0
    K3 = 30.0
    K4 = 40.0

    t0 = 0.0
    t1 = 60.0
    t = collect(t0:0.01:t1)

    dq_values = -dQ.(K1, φ, c, t)
    p = plot(t, dq_values)

    dq_values = -dQ.(K2, φ, c, t)
    plot!(t, dq_values)

    dq_values = -dQ.(K3, φ, c, t)
    plot!(t, dq_values)

    dq_values = -dQ.(K4, φ, c, t)
    plot!(t, dq_values)

    plot!(ylabel=L"$- \frac{\mathrm{d} Q_{1,2} (t)}{\mathrm{d} t}$")

    plot!(legend = false)
    plot!(frame = :box)
    plot!(xlims = [t0,t1])
    plot!(ylims=[-0.1,1.1])
    plot!(xticks = (t0:10:t1,[]))
    return p
end

function plot_ddq(c, φ)
    K1 = 10.0
    K2 = 20.0
    K3 = 30.0
    K4 = 40.0

    t0 = 0.0
    t1 = 60.0
    t = collect(t0:0.01:t1)

    ddq_values = ddQ.(K1, φ, c, t)
    p = plot(t, ddq_values)

    ddq_values = ddQ.(K2, φ, c, t)
    plot!(t, ddq_values)

    ddq_values = ddQ.(K3, φ, c, t)
    plot!(t, ddq_values)

    ddq_values = ddQ.(K4, φ, c, t)
    plot!(t, ddq_values)

    # plot the Gaussians for comparison
    ddq_values = normal.(K1, φ, c, t)
    plot!(t, ddq_values, color=:black,ln=2,style=:dot,label=false)

    ddq_values = normal.(K2, φ, c, t)
    plot!(t, ddq_values, color=:black,ln=2,style=:dot,label=false)

    ddq_values = normal.(K3, φ, c, t)
    plot!(t, ddq_values, color=:black,ln=2,style=:dot,label=false)

    ddq_values = normal.(K4, φ, c, t)
    plot!(t, ddq_values, color=:black,ln=2,style=:dot,label=false)

    plot!(xlabel=L"$t$")
    plot!(ylabel=L"$\frac{\mathrm{d}^2 Q_{1,2} (t)}{\mathrm{d} t^2}$")

    plot!(legend = false)
    plot!(frame = :box)
    plot!(xlims = [t0,t1])
    plot!(ylims=[-0.025,0.3])
    plot!(xticks = t0:10:t1)
    plot!(yticks = 0:0.1:0.3)
    return p
end

function non_stochastic_eqs(c, φ)
    p1 = plot_q(c, φ)
    p2 = plot_dq(c, φ)
    p3 = plot_ddq(c, φ)

    p = plot(p1, p2, p3, layout = (3, 1), left_margin=2mm, size=(600,500))
    savefig(p, "dequeuing_example.pdf") 

    return p
end

function test_sde(c, φ)
    tspan = (0.0, 60.0)
    t0 = tspan[1]
    t1 = tspan[2]

    function solve_sde(K, c, φ)
        u0 = K
        p = (-1/φ, sqrt(c))

        drift_fun(u, p, t) = p[1] * sign(u)
        diffusion_fun(u, p, t) = p[2] * tanh(u)

        prob = SDEProblem(drift_fun, diffusion_fun, u0, tspan, p)
        return solve(prob, EM(), dt=0.0125, adaptive=false)

        # f(u,p,t) = -sign(u) / (φ - c)
        # g(u,p,t) = 0.0
        # u0 = K

        # prob = SDEProblem(f, g, u0, tspan)

        # rate_func(u, p, t) = sqrt(c)
        # function jump_affect!(integrator)
        #     integrator.u += tanh(integrator.u) / sqrt(c)
        # end
        # jump = ConstantRateJump(rate_func, jump_affect!)
        # jump_prob = JumpProblem(prob, Direct(), jump)

        # return solve(jump_prob, SRIW1(), dt=0.01, adaptive=false)
    end

    function avg_solve_sde(K, c, φ)
        n_threads = Threads.nthreads()

        dt = (t1 - t0) / 1000.0
        T = collect(t0:dt:t1)
        U = zeros(Float64,length(T),n_threads)

        n_trials = 50000
        Threads.@threads for thread_id in 1:n_threads
            for _ in thread_id:n_threads:n_trials
                sol = solve_sde(K, c, φ)
                U[:,thread_id] += (1.0 / n_trials) .* sol.(T)
            end
        end

        return T, sum(U,dims=2)
    end

    K1 = 10.0
    K2 = 20.0
    K3 = 30.0
    K4 = 40.0

    T1, U1 = avg_solve_sde(K1,c,φ)
    T2, U2 = avg_solve_sde(K2,c,φ)
    T3, U3 = avg_solve_sde(K3,c,φ)
    T4, U4 = avg_solve_sde(K4,c,φ)

    U1_model = Q.(K1,φ,c,T1)
    U2_model = Q.(K2,φ,c,T2)
    U3_model = Q.(K3,φ,c,T3)
    U4_model = Q.(K4,φ,c,T4)

    p1 = plot(T1,U1, label = L"K = 10")
    plot!(T2[:],U2, label = L"K = 20")
    plot!(T3[:],U3, label = L"K = 30")
    plot!(T4[:],U4, label = L"K = 40")

    plot!(T1,U1_model,color=:black,ln=2,style=:dot,label=false)
    plot!(T2,U2_model,color=:black,ln=2,style=:dot,label=false)
    plot!(T3,U3_model,color=:black,ln=2,style=:dot,label=false)
    plot!(T4,U4_model,color=:black,ln=2,style=:dot,label=false)

    plot!(frame = :box)
    plot!(xlims = [t0,t1])
    plot!(ylims=[-5,50])
    plot!(xticks = (t0:10:t1,[]))

    plot!(ylabel=L"$\overline{Q_{1,2} (t)}$")

    function get_derivs(T,U)
        U = (U[2:end] - U[1:(end-1)]) ./ (T[2:end] - T[1:(end-1)])
        T = T[1:(end-1)]

        return T, U
    end

    T1, U1 = get_derivs(T1, U1)
    T2, U2 = get_derivs(T2, U2)
    T3, U3 = get_derivs(T3, U3)
    T4, U4 = get_derivs(T4, U4)

    dU1_model = dQ.(K1,φ,c,T1)
    dU2_model = dQ.(K2,φ,c,T2)
    dU3_model = dQ.(K3,φ,c,T3)
    dU4_model = dQ.(K4,φ,c,T4)

    p2 = plot(T1,-U1)
    plot!(T2[:],-U2)
    plot!(T3[:],-U3)
    plot!(T4[:],-U4)

    plot!(xticks = t0:5:t1)

    plot!(T1,-dU1_model,color=:black,ln=2,style=:dot,label=false)
    plot!(T2,-dU2_model,color=:black,ln=2,style=:dot,label=false)
    plot!(T3,-dU3_model,color=:black,ln=2,style=:dot,label=false)
    plot!(T4,-dU4_model,color=:black,ln=2,style=:dot,label=false)

    plot!(frame = :box)
    plot!(legend = false)
    plot!(xlims = [t0,t1])
    plot!(ylims=[-0.1,1.1])
    plot!(xticks = (t0:10:t1))

    plot!(xlabel=L"$t$")
    plot!(ylabel=L"$- \frac{\mathrm{d} \overline{Q_{1,2} (t)}}{\mathrm{d} t}$")

    p = plot(p1,p2,layout=(2,1), left_margin=4mm, size=(600,375))
    savefig(p, "dequeuing_example_stochastic.pdf") 
    return p
end

