module SurrogateModelVAE

include("SurrogateModelProtocol.jl")
using .SurrogateModelProtocol

using Flux
using Random
using Zygote
using MLUtils
using Functors
using Statistics
using Distributions
using LinearAlgebra
using QuantumSavory
using DocStringExtensions
using ConcurrentSim, ResumableFunctions

export 
    SurrogateProtocolVAE
abstract type AbstractSurrogateModel end

"""
$TYPEDEF

A protocol simulating the behavior of a quantum network, trained using a 
variational autoencoder (VAE) architecture. The VAE is designed to learn a 
latent representation of the protocol's behavior based on training data 
consisting of time-series data of observables from the quantum network 
simulations. The surrogate protocol can then be used to generate new time-series 
data that mimics the behavior of the original protocol, allowing for faster 
simulations and analysis.

$FIELDS
"""
@kwdef struct VariationalAutoencoder <: AbstractSurrogateModel
    """Dimensionality of the latent space representation."""
    latent_dim::Int = 1
    """Number of latent layers in the encoder and decoder chains of the VAE."""
    latent_layers::Int = 1
    """Encoder chain of the VAE, responsible for encoding time-series data into a latent representation."""
    encoders::Tuple{Vararg{Chain}} 
    """Decoder chain of the VAE, responsible for generating time-series data from the latent representation."""
    decoders::Tuple{Vararg{Chain}}
    """Weighting factor for the KL divergence term in the VAE loss function."""
    β::Union{Float64,Vector{Float64}} = 1.0
end

Functors.@functor VariationalAutoencoder (encoders, decoders)

function load_variational_autoencoder_parameters(saved_model::Any)
    throw(MethodError(load_variational_autoencoder_parameters, (saved_model)))
end

function load_training_data(training_data::Any)
    throw(MethodError(load_training_data, (training_data)))
end

function default_encoder_final_network(num_inputs::Int, latent_dim::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(num_inputs, hidden_layer_size, swish), 
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end

function default_encoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(latent_dim, hidden_layer_size, swish), 
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end

function default_encoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32)
    encoders = Vector{Chain}(undef, latent_layers)
    encoders[1] = default_encoder_final_network(num_inputs, latent_dim, hidden_layer_size)

    if latent_layers > 1
        encoders[2:end] = [default_encoder_mid_network(latent_dim, hidden_layer_size) for _ in 1:(latent_layers-1)]
    end

    return Tuple(encoders)
end

function default_decoder_final_network(num_inputs::Int, latent_dim::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(latent_dim, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, num_inputs)
    ) |> f64
end

function default_decoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(latent_dim, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 2*latent_dim)
    ) |> f64
end

function default_decoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32)
    decoders = Vector{Chain}(undef, latent_layers)
    decoders[1] = default_decoder_final_network(num_inputs, latent_dim, hidden_layer_size)

    if latent_layers > 1
        decoders[2:end] = [default_decoder_mid_network(latent_dim, hidden_layer_size) for _ in 1:(latent_layers-1)]
    end

    return Tuple(decoders)
end

function SurrogateModelProtocolVAE(
    simulation::Simulation; 
    training_data_path::Union{String, Nothing} = nothing, 
    saved_model::Union{String, Nothing} = nothing, 
    encoders::Union{Tuple{Chain}, Nothing} = nothing, 
    decoders::Union{Tuple{Chain}, Nothing} = nothing, 
    single_shot = false, 
    simulation_window_size::Int = 1, 
    latent_dim::Int = 1, 
    latent_layers::Int = 1,
    β::Union{Float64,Vector{Float64}} = 1.0,
    kwargs...)
    training_data = zeros(Float64,0,0)
    if !isnothing(training_data_path)
        raw_training_data = load_training_data(training_data_path)
        reshaped_training_data = Vector{Matrix{Float64}}(undef, length(raw_training_data))

        for i in eachindex(raw_training_data)
            total_rows = size(raw_training_data[i], 1)
            total_windows = total_rows - simulation_window_size

            raw_mat = raw_training_data[i]
            reshaped_matrix = nothing
            if !single_shot
                t_diff_mat = zeros(Float64, size(raw_mat)[1]-1, size(raw_mat)[2])
                t_diff_mat[:,1] = diff(raw_mat[:,1])
                t_diff_mat[:,2:end] = raw_mat[1:(end-1),2:end]
                reshaped_matrix = [vec(t_diff_mat[j:(j + simulation_window_size - 1), :]) for j in 1:total_windows]
            else
                reshaped_matrix = [vec(raw_mat[j:(j + simulation_window_size - 1), :]) for j in 1:total_windows]
            end

            reshaped_training_data[i] = collect(transpose(reduce(hcat, reshaped_matrix)))
        end

        reshaped_training_data = collect(transpose(reduce(vcat,reshaped_training_data)))
        training_data = Float64.(reshaped_training_data)
    end

    if !isnothing(saved_model)
        autoencoder_parameters = load_variational_autoencoder_parameters(saved_model)
        encoders = Vector{Chain}(undef, length(autoencoder_parameters.encoders))
        decoders = Vector{Chain}(undef, length(autoencoder_parameters.decoders))

        for i in eachindex(autoencoder_parameters.encoders)
            encoders[i] = Chain([layer_parameter(layer) for layer in autoencoder_parameters.encoders[i].layers]...)
        end

        for i in eachindex(autoencoder_parameters.decoders)
            decoders[i] = Chain([layer_parameter(layer) for layer in autoencoder_parameters.decoders[i].layers]...)
        end

        latent_dim = size(decoders[1][1].weight)[2]
        encoders = Tuple(encoders)
        decoders = Tuple(decoders)
    else
        if isnothing(encoders)
            num_inputs = maximum([1,size(training_data)[1]])
            encoders = default_encoder_network(num_inputs,latent_dim,latent_layers)
        end

        if isnothing(decoders)
            num_inputs = maximum([1,size(training_data)[1]])
            decoders = default_decoder_network(num_inputs,latent_dim,latent_layers)
        end
    end

    model = VariationalAutoencoder(;
        latent_dim = latent_dim,
        latent_layers = latent_layers,
        encoders = encoders,
        decoders = decoders,
        β = β
    )

    return SurrogateModelProtocol(;
        simulation = simulation,
        training_data_path = training_data_path,
        training_data = training_data,
        simulation_window_size = simulation_window_size, 
        model = model,
        kwargs...
    )
end

function sample_latent(μ, σ)
    ϵ = Flux.randn_like(μ, size(μ))
    return μ .+ σ .* ϵ
end

function _encode(encoders::Tuple{Vararg{Chain}}, x, latent_dim, num_latent_layers)
    batch_size = size(x, 2)
    
    μ_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)
    σ_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)
    logσ²_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)
    z_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)

    enc_out = encoders[1](x)
    μ_buf[:, 1, :] = enc_out[1:latent_dim, :]
    logσ²_buf[:, 1, :] = enc_out[(latent_dim+1):end, :]
    σ_buf[:, 1, :] = exp.(logσ²_buf[:, 1, :] .* 0.5f0)
    z_buf[:, 1, :] = sample_latent(μ_buf[:, 1, :], σ_buf[:, 1, :])

    for i in 2:num_latent_layers
        enc_out = encoders[i](z_buf[:, i-1, :])
        μ_buf[:, i, :] = enc_out[1:latent_dim, :]
        logσ²_buf[:, i, :] = enc_out[(latent_dim+1):end, :]
        σ_buf[:, i, :] = exp.(logσ²_buf[:, i, :] .* 0.5f0)
        z_buf[:, i, :] = sample_latent(μ_buf[:, i, :], σ_buf[:, i, :])
    end

    return copy(μ_buf), copy(σ_buf), copy(logσ²_buf), copy(z_buf)
end

function _decode(decoders::Tuple{Vararg{Chain}}, z, latent_dim, num_latent_layers)
    batch_size = size(z, 3)
    
    μ_buf = Zygote.Buffer(z, eltype(z), latent_dim, num_latent_layers, batch_size)
    σ_buf = Zygote.Buffer(z, eltype(z), latent_dim, num_latent_layers, batch_size)
    logσ²_buf = Zygote.Buffer(z, eltype(z), latent_dim, num_latent_layers, batch_size)

    μ_buf[:, num_latent_layers, :] = Flux.zeros_like(z, (latent_dim, batch_size))
    σ_buf[:, num_latent_layers, :] = Flux.ones_like(z, (latent_dim, batch_size))
    logσ²_buf[:, num_latent_layers, :] = Flux.zeros_like(z, (latent_dim, batch_size))

    for i in 2:num_latent_layers
        idx = num_latent_layers - i + 1
        dec_out = decoders[idx+1](z[:, idx+1, :])
        μ_buf[:, idx, :] = dec_out[1:latent_dim, :]
        logσ²_buf[:, idx, :] = dec_out[(latent_dim+1):end, :]
        σ_buf[:, idx, :] = exp.(logσ²_buf[:, idx, :] .* 0.5f0)
    end

    x̂ = decoders[1](z[:, 1, :])
    return copy(μ_buf), copy(σ_buf), copy(logσ²_buf), x̂
end

function vae_loss(model::VariationalAutoencoder, input_weights, x)
    batch_size = size(x, 2)
    latent_dim = size(model.decoders[1][1].weight, 2)
    num_latent_layers = length(model.decoders)

    μ_enc, σ_enc, logσ²_enc, z = _encode(model.encoders, x, latent_dim, num_latent_layers)
    μ_dec, σ_dec, logσ²_dec, x̂ = _decode(model.decoders, z, latent_dim, num_latent_layers)

    recon_loss = 0.5f0 .* sum(input_weights .* ((x .- x̂).^2))   
    kl_loss = 0.5f0 .* sum(logσ²_dec .- logσ²_enc .+ (σ_enc.^2 .+ (μ_enc .- μ_dec).^2) ./ σ_dec.^2 .- 1.0f0)

    return (recon_loss + model.β * kl_loss) / batch_size
end

function _train!(protocol::SurrogateModelProtocol, model::VariationalAutoencoder, β::Number; print_log::Bool = true)
    model_train_device = model |> protocol.device
    opt_state = Flux.setup(protocol.optimiser, model_train_device)
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = Float64.(protocol.training_data) |> protocol.device
    
    loader = Flux.DataLoader(
        training_data_device, 
        batchsize = batchsize_device, 
        shuffle = false,
        parallel = true
    )

    input_weights = [1.0 / var(protocol.training_data[i,:]) for i in 1:size(protocol.training_data,1)]
    input_weights[input_weights .> 1.0E9] .= 0.0 # ignore nearly deterministic inputs
    input_weights_device = input_weights |> protocol.device

    training_log = VAETrainingLog(
        zeros(Float64,protocol.epochs),
        zeros(Float64,protocol.epochs)
    )

    if print_log; println("Training VAE... (β = $β)") end
    for epoch in 1:protocol.epochs
        epoch_loss = 0f0
        total_grad_norm = 0f0

        if epoch % 100 == 0 && protocol.shuffle
            loader = Flux.DataLoader(
                shuffleobs(training_data_device), 
                batchsize = batchsize_device, 
                shuffle = false,
                parallel = true
            )
        end

        for x_batch in loader
            loss, grads = Flux.withgradient(model_train_device) do m
                vae_loss(m, input_weights_device, x_batch)
            end

            raw_gradient_arrays = Optimisers.trainables(grads[1])
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model_train_device, grads[1])
            epoch_loss += loss
            total_grad_norm += batch_grad_norm
        end

        training_log.loss[epoch] = epoch_loss / length(loader)
        training_log.loss_grad_norm[epoch] = total_grad_norm / length(loader)

        if epoch % 5 == 0 || epoch == 1
            average_loss = training_log.loss[epoch]
            average_grad_norm = training_log.loss_grad_norm[epoch]
            if print_log; println("Epoch $epoch | Average Loss: $average_loss | Grad Norm: $average_grad_norm") end
        end
    end
    if print_log; println("Training complete!") end

    for i in eachindex(model.encoders)
        Flux.loadmodel!(model.encoders[i], model_train_device.encoders[i])
    end

    for i in eachindex(model.decoders)
        Flux.loadmodel!(model.decoders[i], model_train_device.decoders[i])
    end

    return training_log
end

function _train!(protocol::SurrogateProtocolVAE, model::VariationalAutoencoder, β::Vector; print_log::Bool = true)
    training_log = nothing
    for i in eachindex(β)
        training_log = _train!(protocol, model, β[i]; print_log = print_log)
    end

    return training_log
end

function SurrogateModelProtocol.train!(protocol::SurrogateModelProtocol, model::VariationalAutoencoder; print_log::Bool = true)
    return _train!(protocol, model, model.β; print_log = print_log)
end

function decode(model::VariationalAutoencoder, z)
    num_latent_layers = length(model.decoders)

    for i in 1:(num_latent_layers-1)
        dec_out = model.decoders[num_latent_layers-i+1](z)
        z = sample_latent(dec_out[1:model.latent_dim], exp.(dec_out[(model.latent_dim+1):end] .* 0.5f0))
    end

    return protocol.decoders[1](z) |> protocol.device
end

function encode(model::VariationalAutoencoder, x)
    num_latent_layers = length(model.encoders)

    enc_out = model.encoders[1](x)
    z = sample_latent(enc_out[1:model.latent_dim,:], exp.(enc_out[(model.latent_dim+1):end,:] .* 0.5f0))
    
    for i in 2:num_latent_layers
        enc_out = model.encoders[i](z)
        z = sample_latent(enc_out[1:model.latent_dim,:], exp.(enc_out[(model.latent_dim+1):end,:] .* 0.5f0))
    end

    return z
end

function SurrogateModelProtocol._eval(protocol::SurrogateModelProtocol, model::VariationalAutoencoder, n_samples::Int)
    window_size = size(protocol.training_data,1)
    sim_log = zeros(Float64,n_samples,window_size)

    for i in 1:n_samples
        z = rand(Normal(0.0,1.0),model.latent_dim)
        sim_log[i,:] = decode(model,z)
    end

    return sim_log
end

struct VariationalAutoencoderParameters
    encoders::Vector{SurrogateModelProtocol.ChainParameters}
    decoders::Vector{SurrogateModelProtocol.ChainParameters}
end

function variational_autoencoder_parameters(model::VariationalAutoencoder)
    return VariationalAutoencoderParameters(
        [AbstractSurrogateProtocol.chain_parameters(encoder) for encoder in model.encoders],
        [AbstractSurrogateProtocol.chain_parameters(decoder) for decoder in model.decoders]
    )
end

include("SurrogateProtocolVAE_FileIO.jl")
include("SurrogateProtocolVAE_HDF5.jl")

end

