# module SurrogateProtocolVAE_FileIO

# using SurrogateProtocolVAE
# using ConcurrentSim, ResumableFunctions
using QuantumSavory
using QuantumSavory.ProtocolZoo
import FileIO

function _set_metadata_default_values(metadata::Dict{String,Any})
    if !haskey(metadata, "reference_state")
        metadata["reference_state"] = "bell_pair"
    end

    if !haskey(metadata, "log_format")
        metadata["log_format"] = "pauli_observables"
    end

    if !haskey(metadata, "simulation_mode")
        metadata["simulation_mode"] = "stateful"
    end

    if !haskey(metadata, "description")
        metadata["description"] = ""
    end

    metadata["simulator"] = "Variational Autoencoder Surrogate"

    return metadata
end

function FileIO.save(file_name::String, prot::SurrogateProtocolVAE; metadata::Union{Dict{String,Any}, NamedTuple, Nothing} = nothing)
    if isnothing(metadata)
        metadata = Dict{String,Any}()
    elseif isa(metadata, NamedTuple)
        metadata = Dict(string(k) => v for (k, v) in pairs(metadata))
    end
    
    metadata = _set_metadata_default_values(metadata)
    FileIO.save(FileIO.query(file_name), prot; metadata)
end

function load_training_data(training_data_path::String)
    if isfile(training_data_path)
        consumer_log = EntanglementConsumerLog(training_data_path)
        training_data = hcat(consumer_log.simulation_log.time, consumer_log.simulation_log.state')
        return [training_data]
    else
        file_names = filter(x -> isfile(x), readdir(training_data_path, join=true))
        training_data = Vector{Matrix{Float64}}(undef, length(file_names))

        for sample_id in eachindex(file_names)
            consumer_log = EntanglementConsumerLog(file_names[sample_id])

            if !isnothing(consumer_log)
                training_data[sample_id] = hcat(consumer_log.simulation_log.time, consumer_log.simulation_log.state')
            else
                training_data[sample_id] = zeros(Float64,0,0)
            end
        end
    end

    return filter(x -> !isempty(x), training_data)
end

function load_variational_autoencoder_parameters(saved_model::String)
    return load_variational_autoencoder_parameters(FileIO.query(saved_model))
end

function default_encoder_final_network(num_inputs::Int, latent_dim::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(num_inputs, hidden_layer_size, swish), 
        Dense(hidden_layer_size, hidden_layer_size, sigmoid),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, sigmoid),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end

function default_encoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(latent_dim, hidden_layer_size, swish), 
        Dense(hidden_layer_size, hidden_layer_size, sigmoid),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, sigmoid),
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
    # return Chain(
    #     Dense(latent_dim, hidden_layer_size, swish),
    #     Dense(hidden_layer_size, hidden_layer_size, sigmoid),
    #     Dense(hidden_layer_size, hidden_layer_size, swish),
    #     Dense(hidden_layer_size, hidden_layer_size, sigmoid),
    #     Dense(hidden_layer_size, hidden_layer_size, swish),
    #     Dense(hidden_layer_size, num_inputs)
    # ) |> f64

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
    # return Chain(
    #     Dense(latent_dim, hidden_layer_size, swish),
    #     Dense(hidden_layer_size, hidden_layer_size, sigmoid),
    #     Dense(hidden_layer_size, hidden_layer_size, swish),
    #     Dense(hidden_layer_size, hidden_layer_size, sigmoid),
    #     Dense(hidden_layer_size, hidden_layer_size, swish),
    #     Dense(hidden_layer_size, 2*latent_dim)
    # ) |> f64

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

function SurrogateProtocolVAE(
    simulation::Simulation; 
    training_data_path::Union{String, Nothing} = nothing, 
    saved_model::Union{String, Nothing} = nothing, 
    encoders::Union{Tuple{Chain}, Nothing} = nothing, 
    decoders::Union{Tuple{Chain}, Nothing} = nothing, 
    single_shot = false, 
    simulation_window_size::Int = 1, 
    latent_dim::Int = 1, 
    latent_layers::Int = 1,
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

    return SurrogateProtocolVAE(;
        simulation = simulation,
        training_data_path = training_data_path,
        training_data = training_data,
        simulation_window_size = simulation_window_size, 
        latent_dim = latent_dim, 
        decoders = decoders,
        encoders = encoders,
        kwargs...
    )
end

# function SurrogateProtocolVAE(simulation::Simulation; trained_model_path::String)
    # return SurrogateProtocolVAE(simulation; trained_model_path = FileIO.query(trained_model_path))
# end

# end

