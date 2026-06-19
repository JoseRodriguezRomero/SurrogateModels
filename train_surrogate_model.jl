using ConcurrentSim, Plots, KernelDensity, Distributions, Flux
using FileIO

include("SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

# sim = Simulation()
# surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim; 
#     training_data_path = "sequence_data/linear_17_swap_prob=0.9.h5", 
#     single_shot = true,
#     simulation_window_size = 1,
#     latent_dim = 2, 
#     batchsize = 128, 
#     epochs = 500, 
#     β = [0.01, 1.0],
#     optimiser = Adam(;eta = 1.0E-4, beta = (0.8,0.999))
# )
# SurrogateProtocolVAE.train!(surrogate)
# save("trained_surrogates/sequence/linear_17_swap_prob=0.9.h5", surrogate)

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    training_data_path = "saved_data/F_0.95/",
    simulation_window_size = 1,
    latent_dim = 9,
    batchsize = 128, 
    epochs = 2500, 
    β = [0.01, 1.0],
    optimiser = Adam(;eta = 1.0E-4, beta = (0.8,0.999))
)
SurrogateProtocolVAE.train!(surrogate)
save("trained_surrogates/quantum_savory/F_0.95.h5", surrogate)

