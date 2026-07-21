using ConcurrentSim, Flux, FileIO
using DataFrames, CSV

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

device = cpu_device()

try
    using CUDA, cuDNN
    global device = gpu_device()
catch
    println("No CUDA support detected! Using CPU as compute device.")
end

function save_log_to_csv(file_name, training_log)
    df = DataFrame(:loss => training_log.loss, :grad_norm => training_log.loss_grad_norm)
    CSV.write(file_name, df)
end

base_dir = @__DIR__() * "/"
all_swap_prob = [0.8, 0.9]

for swap_prob in all_swap_prob
    for latent_layers in 1:4
        if !ispath("../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers")
            mkpath("trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers")
        end
    end

    for latent_layers in 1:4
        training_data_path = base_dir * "/../sequence_data/linear_17_swap_prob=$swap_prob.h5"
        save_path = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers/linear_17_swap_prob=$swap_prob.h5"
        training_log_save_path = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob/latent_layers_$latent_layers/linear_17_swap_prob=$swap_prob.csv"

        sim = Simulation()
        surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
            single_shot = true,
            training_data_path = training_data_path,
            saved_model = save_path,
            simulation_window_size = 1,
            latent_dim = 2,
            latent_layers = latent_layers,
            batchsize = 2048,
            epochs = 2500, 
            β = [0.1],
            optimiser = Adam(;eta = 1.0E-3, beta = (0.999,0.9999)),
            device = device
        )
        training_log = SurrogateProtocolVAE.train!(surrogate)
        save(save_path, surrogate)
        save_log_to_csv(training_log_save_path, training_log)

        try
            for i in 1:5
                CUDA.reclaim()
            end
        catch
        end
    end
end

