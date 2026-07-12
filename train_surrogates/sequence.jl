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
all_swap_prob = [0.8, 0.9]

for swap_prob in all_swap_prob
    for latent_layers in 1:4
        if !ispath("../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers")
            mkpath("trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers")
        end
    end
end

for swap_prob in all_swap_prob
    println("Begining linear_17_swap_prob=$swap_prob")
    @threads for latent_layers in 1:4
        training_data_path = base_dir * "/../sequence_data/linear_17_swap_prob=$swap_prob.h5"
        save_path = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers/linear_17_swap_prob=$swap_prob.h5"
        training_log_save_path = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers/linear_17_swap_prob=$swap_prob.csv"

        sim = Simulation()
        surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
            single_shot = true,
            training_data_path = training_data_path,
            # saved_model = save_path,
            simulation_window_size = 1,
            latent_dim = 2,
            latent_layers = latent_layers,
            batchsize = 256,
            epochs = 7500, 
            β = [0.1],
            optimiser = Adam(;eta = 1.0E-4, beta = (0.9,0.999))
        )
        training_log = SurrogateProtocolVAE.train!(surrogate; print_log = false)
        save(save_path, surrogate)
        save_log_to_csv(training_log_save_path, training_log)
    end
    println("Finished linear_17_swap_prob=$swap_prob")
end

for swap_prob in all_swap_prob
    for latent_layers in 1:4
        training_log_save_path = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers/linear_17_swap_prob=$swap_prob.csv"
        training_plot_save_file_name = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers/ELBO_training"

        plot_train_loss(training_log_save_path, training_plot_save_file_name)
    end
end

