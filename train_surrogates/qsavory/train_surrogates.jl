using ConcurrentSim, Flux, FileIO
using DataFrames, CSV, HDF5

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

function read_data(file_path::String)
    sim_log = EntanglementConsumerLog(file_path)
    training_data = hcat(
        sim_log.simulation_log.time, 
        sim_log.simulation_log.state[1,:], 
        sim_log.simulation_log.state[2,:]
    )
    training_data = hcat(
        reshape(diff(training_data[:,1]), :, 1),
        training_data[2:end,2:end]
    )

    return collect(transpose(training_data))
end

function read_all_data(base_dir::String)
    files = filter(f -> endswith(f, ".h5"), readdir(base_dir))
    training_data = zeros(Float64,3,0)

    for file in files
        training_data = hcat(training_data, read_data(base_dir * file))
    end

    return training_data
end

function train_all_surrogates(model::GenerativeModelProtocols.AbstractGenerativeModel, base_dir::String)
    for fidelity in [0.75, 0.95]
        if !ispath(base_dir * "/F_$fidelity")
            mkpath(base_dir * "/F_$fidelity")
        end

        training_data = read_all_data(@__DIR__() * "/../../qsavory_data/reference_data/F_$fidelity/")
        save_path = base_dir * "/F_$fidelity/F_$fidelity.h5"
        training_log_save_path = base_dir * "/F_$fidelity/F_$fidelity.csv"

        surrogate = GenerativeModelProtocol(deepcopy(model), training_data;
            batchsize = 8192,
            epochs = 10,
            optimiser = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
            device = device
        )
        training_log = train!(surrogate)
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
        model = GenerativeModelProtocols.VariationalAutoencoder(3, 3, latent_layers; β = [0.1])
        base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/variational_autoencoder/latent_layers_$latent_layers"
        train_all_surrogates(model, base_dir)
    end
end

function train_all_dm_surrogates()
    model = GenerativeModelProtocols.DiffusionModel(3)
    base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/diffusion_model"
    train_all_surrogates(model, base_dir)
end

function train_all_surrogates()
    train_all_vae_surrogates()
    train_all_dm_surrogates()
end

train_all_surrogates()

