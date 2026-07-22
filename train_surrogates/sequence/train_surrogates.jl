using Flux
using DataFrames
using FileIO, HDF5, CSV

using GenerativeModelProtocols

using QuantumSavory
using QuantumSavory.ProtocolZoo

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

base_dir = @__DIR__()

function train_all_surrogates(model::GenerativeModelProtocols.AbstractGenerativeModel, base_dir::String)
    for swap_prob in [0.8, 0.9]
        if !ispath(base_dir)
            mkpath(base_dir)
        end

        training_data_path = @__DIR__() * "/../../sequence_data/reference_data/linear_17_swap_prob=$swap_prob.h5"
        save_path = base_dir * "/linear_17_swap_prob=$swap_prob.h5"
        training_log_save_path = base_dir * "/linear_17_swap_prob=$swap_prob.csv"

        sim_log = EntanglementConsumerLog(training_data_path)
        training_data = hcat(sim_log.simulation_log.time, sim_log.simulation_log.state[:])
        training_data = collect(transpose(training_data))

        surrogate = GenerativeModelProtocol(deepcopy(model), training_data;
            batchsize = 2048,
            epochs = 1500,
            optimiser = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
            device = device
        )

        t₀ = time()
        training_log = train!(surrogate)
        t₁ = time()
        println("Training took: $(t₁ - t₀) seconds")

        save(save_path, surrogate)
        save_log_to_csv(training_log_save_path, training_log)

        try
            for _ in 1:5
                CUDA.reclaim()
            end
        catch
        end
    end
end

function train_all_vae_surrogates()
    for latent_layers in 1:4
        model = GenerativeModelProtocols.VariationalAutoencoder(2, 2, latent_layers; β = [0.2])
        base_dir = @__DIR__() * "/../../sequence_data/trained_surrogates/variational_autoencoder/latent_layers_$latent_layers"
        train_all_surrogates(model, base_dir)
    end
end

function train_all_dm_surrogates()
    model = GenerativeModelProtocols.DiffusionModel(2)
    base_dir = @__DIR__() * "/../../sequence_data/trained_surrogates/diffusion_model"
    train_all_surrogates(model, base_dir)
end

function train_all_surrogates()
    train_all_vae_surrogates()
    # train_all_dm_surrogates()
end

train_all_surrogates()

