using ConcurrentSim, Flux, FileIO
using DataFrames, CSV, Plots
using Base.Threads

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

function save_log_to_csv(file_name, training_log)
    df = DataFrame(:loss => training_log.loss, :grad_norm => training_log.loss_grad_norm)
    CSV.write(file_name, df)
end

function plot_train_loss(file_name::String, plot_save_file_name::String)
    data = CSV.read(file_name, DataFrame)
    p = plot(data.loss, xlabel="Epochs", xaxis=:log, yaxis=:log, frame=:box, label="-ELBO")
    plot!(data.grad_norm, xlabel="Epochs", xaxis=:log, yaxis=:log, frame=:box, label="-ELBO norm")
    savefig(p, plot_save_file_name * ".pdf")
    savefig(p, plot_save_file_name * ".svg")
    savefig(p, plot_save_file_name * ".png")
end

base_dir = @__DIR__() * "/"

for fidelity in [0.75, 0.95]
    for latent_layers in 1:4
        if !ispath("../trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers")
            mkpath("trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers")
        end
    end
end

for fidelity in [0.75, 0.95]
    @threads for latent_layers in 1:4
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
            batchsize = 256,
            epochs = 2500, 
            β = [0.1],
            optimiser = Adam(;eta = 1.0E-3, beta = (0.95,0.999))
        )
        training_log = SurrogateProtocolVAE.train!(surrogate; print_log = false)
        save(save_path, surrogate)
        save_log_to_csv(training_log_save_path, training_log)
    end
end

for fidelity in [0.75, 0.95]
    for latent_layers in 1:4
        training_log_save_path = base_dir * "../trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers/F_$fidelity.csv"
        training_plot_save_file_name = base_dir * "../trained_surrogates/quantum_savory/F_$fidelity/latent_layers_$latent_layers/ELBO_training"

        plot_train_loss(training_log_save_path,training_plot_save_file_name)
    end
end

