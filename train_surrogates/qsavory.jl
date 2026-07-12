using ConcurrentSim, Flux, FileIO
using DataFrames, CSV
using Base.Threads

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

function save_log_to_csv(file_name, training_log)
    df = DataFrame(:loss => training_log.loss, :grad_norm => training_log.loss_grad_norm)
    CSV.write(file_name, df)
end

base_dir = @__DIR__() * "/"

for fidelity in [0.75, 0.95]
    for latent_layers in 1:4
        if !ispath("../trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers")
            mkpath("trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers")
        end
    end

    for latent_layers in 1:4
        training_data_path = base_dir * "../saved_data/F_$fidelity/"
        save_path = base_dir * "../trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers/F_$fidelity.h5"
        training_log_save_path = base_dir * "../trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers/F_$fidelity.csv"

        sim = Simulation()
        surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
            training_data_path = training_data_path,
            # saved_model = save_path,
            simulation_window_size = 1,
            latent_dim = 3,
            latent_layers = latent_layers,
            batchsize = 512,
            epochs = 5000, 
            β = [0.1],
            optimiser = Adam(;eta = 1.0E-3, beta = (0.95,0.999))
        )
        training_log = SurrogateProtocolVAE.train!(surrogate)
        save(save_path, surrogate)
        save_log_to_csv(training_log_save_path, training_log)
    end
end

