using ConcurrentSim, Plots, KernelDensity, Distributions, Flux
using FileIO

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    training_data_path = base_dir * "../saved_data/F_0.95/",
    # saved_model = base_dir * "../trained_surrogates/quantum_savory/F_0.95.h5",
    simulation_window_size = 1,
    latent_dim = 3,
    batchsize = 128, 
    epochs = 4500, 
    β = [0.01, 1.0],
    optimiser = Adam(;eta = 1.0E-4, beta = (0.8,0.999))
)
SurrogateProtocolVAE.train!(surrogate)
save(base_dir * "../trained_surrogates/quantum_savory/F_0.95.h5", surrogate)

