using ConcurrentSim, Flux, FileIO

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"
swap_prob = 0.8

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim; 
    training_data_path = base_dir * "../sequence_data/linear_17_swap_prob=$swap_prob.h5",
    saved_model = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.h5",
    single_shot = true,
    simulation_window_size = 1,
    latent_dim = 2, 
    latent_layers = 4,
    batchsize = 128, 
    epochs = 2500, 
    β = [0.1],
    optimiser = Adam(;eta = 1.0E-3, beta = (0.95,0.999))
)
SurrogateProtocolVAE.train!(surrogate)
save(base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.h5", surrogate)

