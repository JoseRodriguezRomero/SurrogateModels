using ConcurrentSim, Plots, KernelDensity, Distributions, Flux
using FileIO

include("SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

# sim = Simulation()
# surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
#     single_shot = true,
#     training_data_path = "sequence_data/linear_17_swap_prob=0.9.h5", 
#     saved_model = "trained_surrogates/sequence/linear_17_swap_prob=0.9.h5"
# )

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    training_data_path = "saved_data/F_0.95/",
    saved_model = "trained_surrogates/quantum_savory/F_0.95.h5"
)

sim_data = surrogate(50000)

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

num_train_samples = size(surrogate.training_data)[2]
bootstrapped_data = zeros(Float64,size(sim_data))
for i in 1:length(bootstrapped_data[:,1])
    bootstrapped_data[i,:] = surrogate.training_data[:,rand(1:num_train_samples)]
end

simulation_window = surrogate.simulation_window_size
p1_range = (0*simulation_window+1):(1*simulation_window)
p2_range = (2*simulation_window+1):(3*simulation_window)
# p2_range = (1*simulation_window+1):(2*simulation_window)

p1 = compare_kde_plot(sim_data[:,p1_range][:], bootstrapped_data[:,p1_range][:])
p2 = compare_kde_plot(sim_data[:,p2_range][:], bootstrapped_data[:,p2_range][:])

plot(p1,p2,layout=(2,1))

