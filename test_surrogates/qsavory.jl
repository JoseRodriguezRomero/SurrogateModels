using ConcurrentSim, Plots, KernelDensity, Distributions, Flux
using FileIO

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    training_data_path = base_dir * "../saved_data/F_0.95/",
    saved_model = base_dir * "../trained_surrogates/quantum_savory/F_0.95.h5"
)

num_vae_samples = 50000
sim_data = surrogate(num_vae_samples)

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

    p = histogram(ground_truth, normalize=:pdf, label="QSavory Simulation")
    p = plot!(plt_range, vae_kde_pdf.(plt_range), label="VAE Surrogate",frame=:box, linewidth=3)
    # plot!(plt_range, ground_truth_kde_pdf.(plt_range), label="QSavory Simulation",frame=:box, linewidth=3)
    
    return p
end

simulation_window = surrogate.simulation_window_size
p1_range = (0*simulation_window+1):(1*simulation_window)
p2_range = (1*simulation_window+1):(2*simulation_window)

p1 = compare_kde_plot(sim_data[:,p1_range][:], surrogate.training_data[p1_range,:][:])
plot!(p1, ylabel="Probability\nDensity")
plot!(p1, xlabel="Time to serve [s]")

p2 = compare_kde_plot(sim_data[:,p2_range][:], surrogate.training_data[p2_range,:][:])
plot!(p2, ylabel="Probability\nDensity")
plot!(p2, xlabel="<X⊗X>")
plot!(p2, legend=false)

plot(p1,p2,layout=(2,1))

