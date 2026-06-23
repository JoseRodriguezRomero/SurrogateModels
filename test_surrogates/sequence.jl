using ConcurrentSim, Plots, KernelDensity

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"
swap_prob = 0.8

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    single_shot = true,
    training_data_path = base_dir *  "../sequence_data/linear_17_swap_prob=$swap_prob.h5", 
    saved_model = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.h5"
)

num_vae_samples = 125000
@time sim_data = surrogate(num_vae_samples)

function compare_kde_plot(vae_sim::Vector, ground_truth::Vector)
    vae_kde = kde(vae_sim)
    vae_kde = InterpKDE(vae_kde)
    vae_kde_pdf(x) = pdf(vae_kde,x)

    ground_truth_kde = kde(ground_truth)
    ground_truth_kde = InterpKDE(ground_truth_kde)
    ground_truth_kde_pdf(x) = pdf(ground_truth_kde,x)

    x0 = minimum(ground_truth)
    x1 = maximum(ground_truth)
    dx = (x1-x0) / 300.0
    plt_range = collect(x0:dx:x1)

    p = histogram(ground_truth, normalize=:pdf, label="SeQUeNCe simulated data")
    p = plot!(plt_range, vae_kde_pdf.(plt_range), label="VAE Surrogate (KDE from synthetic data)",frame=:box, linewidth=3)
    # plot!(plt_range, ground_truth_kde_pdf.(plt_range), label="SeQUeNCe Simulation",frame=:box, linewidth=3)
    
    return p
end

function compare_2d_kde_plot(Δt::Vector, obs::Vector, plot_title::String)
    kde = KernelDensity.kde((Δt, obs))
    kde = KernelDensity.InterpKDE(kde)
    kde_pdf(x,y) = Distributions.pdf(kde,x,y)

    x0 = minimum(Δt)
    x1 = maximum(Δt)
    dx = (x1 - x0) / 400.0

    y0 = minimum(obs)
    y1 = maximum(obs)
    dy = (y1 - y0) / 400.0

    plt = Plots.heatmap(x0:dx:x1, y0:dy:y1, kde_pdf, color=:viridis)
    Plots.plot!(frame=:box)
    Plots.plot!(xlabel = "Time to serve [s]")
    Plots.plot!(ylabel = "Fidelity")
    Plots.plot!(colorbar_title = "\np(Δt,F)")
    Plots.plot!(title = plot_title)
    return plt
end

simulation_window = surrogate.simulation_window_size
p1_range = (0*simulation_window+1):(1*simulation_window)
p2_range = (1*simulation_window+1):(2*simulation_window)

p1 = compare_kde_plot(sim_data[:,p1_range][:], surrogate.training_data[p1_range,:][:])
plot!(p1, ylabel="Probability\nDensity")
plot!(p1, xlabel="Time to serve [ms]")

p2 = compare_kde_plot(sim_data[:,p2_range][:], surrogate.training_data[p2_range,:][:])
plot!(p2, ylabel="Probability\nDensity")
plot!(p2, xlabel="Fidelity")
plot!(p2, legend=false)

p = plot(p1,p2,layout=(2,1))
savefig(p, base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.png")
savefig(p, base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.pdf")
savefig(p, base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.svg")
display(p)

p1 = compare_2d_kde_plot(surrogate.training_data[p1_range,:][:], surrogate.training_data[p2_range,:][:], "2D KDE from SeQUeNCe Data")
p2 = compare_2d_kde_plot(sim_data[:,p1_range][:], sim_data[:,p2_range][:], "2D KDE from VAE Surrogate")

plot!(p1,xlims=(12,80),ylims=(0.43,0.48))
plot!(p2,xlims=(12,80),ylims=(0.43,0.48))

p = plot(p1,p2,layout=(2,1), right_margin = 5mm)
savefig(p, base_dir * "../trained_surrogates/sequence/2d_kde_linear_17_swap_prob=$swap_prob.png")
savefig(p, base_dir * "../trained_surrogates/sequence/2d_kde_linear_17_swap_prob=$swap_prob.pdf")
savefig(p, base_dir * "../trained_surrogates/sequence/2d_kde_linear_17_swap_prob=$swap_prob.svg")
display(p)
