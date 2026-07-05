using ConcurrentSim, Flux, FileIO

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"

fidelity = 0.95

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    training_data_path = base_dir * "../saved_data/F_$fidelity/",
    # saved_model = base_dir * "../trained_surrogates/quantum_savory/F_$fidelity.h5",
    simulation_window_size = 1,
    latent_dim = 3,
    latent_layers = 4,
    batchsize = 256,
    epochs = 2500, 
    β = [0.1],
    optimiser = Adam(;eta = 1.0E-3, beta = (0.95,0.999))
)
SurrogateProtocolVAE.train!(surrogate)
save(base_dir * "../trained_surrogates/quantum_savory/F_$fidelity.h5", surrogate)

