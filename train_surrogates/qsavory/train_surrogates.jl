using Lux, Optimisers
using FileIO, HDF5

using GenerativeModelProtocols

using QuantumSavory
using QuantumSavory.ProtocolZoo

using StatsBase
using Reactant

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

function train_all_surrogates(model::GenerativeModelProtocols.AbstractGenerativeModel, base_dir::String; 
    optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999)), kwargs...)
    for fidelity in [0.75, 0.95]
        if !ispath(base_dir * "/F_$fidelity")
            mkpath(base_dir * "/F_$fidelity")
        end

        training_data = read_all_data(@__DIR__() * "/../../qsavory_data/reference_data/F_$fidelity/")
        save_path = base_dir * "/F_$fidelity/F_$fidelity.h5"
        training_data = training_data[[1, 3], :]

        surrogate = GenerativeModelProtocol(deepcopy(model), training_data)

        train!(surrogate; 
            batchsize = 2048,
            epochs    = 3500,
            optimiser = optimiser,
            device    = reactant_device(),
            kwargs...
        )

        save(save_path, surrogate)
    end
end

function train_all_vae_surrogates()
    for latent_layers in 1:4
        model = GenerativeModelProtocols.VariationalAutoencoder(2, 2, latent_layers)
        base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/variational_autoencoder/latent_layers_$latent_layers"
        train_all_surrogates(model, base_dir; β = 0.1)
    end
end

function train_all_dm_surrogates()
    model = GenerativeModelProtocols.DiffusionModel(2, 300)
    base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/diffusion_model"
    train_all_surrogates(model, base_dir)
end

function train_all_gmm_surrogates()
    model = GenerativeModelProtocols.GaussianMixtureModel(2, 300)
    base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/gaussian_mixture_model"
    train_all_surrogates(model, base_dir)
end

function train_all_gan_surrogates()
    critic_optimiser = Adam(; eta = 1.0E-4, beta = (0.0,0.9))
    vae_optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999))

    model = GenerativeModelProtocols.GenerativeAdversarialNetwork(2, 2)
    base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/generative_adversarial_network"
    train_all_surrogates(model, base_dir;
        optimiser = (critic_optimiser, vae_optimiser),
        β = 0.1, γ_vae = 1.0, γ_wgan = 0.1,
        grad_penalty = true, λ = 10.0, a = 1.0,
        weight_clipping = false, clip_value = 0.1
    )
end

function train_all_nf_surrogates()
    model = GenerativeModelProtocols.NormalizingFlow(2)
    base_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/normalizing_flow"
    train_all_surrogates(model, base_dir)
end

function train_all_surrogates()
    # train_all_vae_surrogates()
    # train_all_dm_surrogates()
    # train_all_gmm_surrogates()
    # train_all_gan_surrogates()
    train_all_nf_surrogates()
end

train_all_surrogates()

