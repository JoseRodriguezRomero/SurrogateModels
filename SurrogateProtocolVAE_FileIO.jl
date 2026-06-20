# module SurrogateProtocolVAE_FileIO

# using SurrogateProtocolVAE
# using ConcurrentSim, ResumableFunctions
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

@kwdef struct EntanglementConsumerLogMetadata
    description::String = ""
    simulator::String = ""
    quantumsavory_metadata::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct EntanglementConsumerLogSimulationLog
    time::Vector{Float64} = zeros(Float64,0)
    state::Matrix{Float64} = zeros(Float64,0,0)
end

@kwdef struct EntanglementConsumerLog
    format_version::UInt64 = UInt64(0)
    format_version_minor::UInt64 = UInt64(0)
    log_format::String = ""
    reference_state::String = ""
    simulation_log::EntanglementConsumerLogSimulationLog = EntanglementConsumerLogSimulationLog()
    metadata::EntanglementConsumerLogMetadata = EntanglementConsumerLogMetadata()
end

function load_entanglement_consumer_log(file_name::String)
    return load_entanglement_consumer_log(FileIO.query(file_name))
end

function load_entanglement_consumer_log(::Any)
    return nothing
end

function load_training_data(training_data_path::String)
    if isfile(training_data_path)
        consumer_log = load_entanglement_consumer_log(training_data_path)
        training_data = hcat(consumer_log.simulation_log.time, consumer_log.simulation_log.state')
        return [training_data]
    else
        file_names = filter(x -> isfile(x), readdir(training_data_path, join=true))
        training_data = Vector{Matrix{Float64}}(undef, length(file_names))

        for sample_id in eachindex(file_names)
            consumer_log = load_entanglement_consumer_log(file_names[sample_id])

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

function default_encoder_network(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32)
    return Chain(
        Dense(num_inputs, hidden_layer_size, swish), 
        Dense(hidden_layer_size, hidden_layer_size, sigmoid),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end


function default_decoder_network(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32)
    return Chain(
        Dense(latent_dim, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, sigmoid),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, num_inputs)
    ) |> f64
end

function SurrogateProtocolVAE(simulation::Simulation; training_data_path::Union{String, Nothing} = nothing, saved_model::Union{String, Nothing} = nothing, encoder_chain::Union{Chain, Nothing} = nothing, decoder_chain::Union{Chain, Nothing} = nothing, single_shot = false, simulation_window_size::Int = 1, latent_dim::Int = 2, kwargs...)
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
        encoder_chain = Chain([layer_parameter(layer) for layer in autoencoder_parameters.encoder.layers]...)
        decoder_chain = Chain([layer_parameter(layer) for layer in autoencoder_parameters.decoder.layers]...)
        latent_dim = size(decoder_chain[1].weight)[2]
    else
        if isnothing(encoder_chain)
            num_inputs = size(training_data)[1]
            encoder_chain = default_encoder_network(num_inputs,latent_dim)
        end

        if isnothing(decoder_chain)
            num_inputs = size(training_data)[1]
            decoder_chain = default_decoder_network(num_inputs,latent_dim)
        end
    end

    return SurrogateProtocolVAE(;
        simulation = simulation,
        training_data_path = training_data_path,
        training_data = training_data,
        simulation_window_size = simulation_window_size, 
        latent_dim = latent_dim, 
        decoder_chain = decoder_chain, 
        encoder_chain = encoder_chain, 
        kwargs...
    )
end

# function SurrogateProtocolVAE(simulation::Simulation; trained_model_path::String)
    # return SurrogateProtocolVAE(simulation; trained_model_path = FileIO.query(trained_model_path))
# end

# end

