using ConcurrentSim, Flux, FileIO

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    training_data_path = base_dir * "../saved_data/F_0.75/",
    # saved_model = base_dir * "../trained_surrogates/quantum_savory/F_0.95.h5",
    simulation_window_size = 1,
    latent_dim = 3,
    batchsize = 128,
    epochs = 3500, 
    β = [0.1],
    optimiser = Adam(;eta = 1.0E-4, beta = (0.8,0.999))
)
SurrogateProtocolVAE.train!(surrogate)
save(base_dir * "../trained_surrogates/quantum_savory/F_0.75.h5", surrogate)

