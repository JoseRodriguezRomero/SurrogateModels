# module SurrogateProtocolVAE_FileIO

# using SurrogateProtocolVAE
# using ConcurrentSim, ResumableFunctions
using QuantumSavory
using QuantumSavory.ProtocolZoo
import FileIO

function FileIO.save(file_name::String, protocol::SurrogateModelProtocol, model::VariationalAutoencoder; metadata::Union{Dict{String,Any}, NamedTuple, Nothing} = nothing)
    if isnothing(metadata)
        metadata = Dict{String,Any}()
    elseif isa(metadata, NamedTuple)
        metadata = Dict(string(k) => v for (k, v) in pairs(metadata))
    end
    
    metadata = _set_metadata_default_values(metadata)
    FileIO.save(FileIO.query(file_name), protocol, model; metadata)
end

function SurrogateModelVAE.load_training_data(training_data_path::String)
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

function SurrogateModelVAE.load_variational_autoencoder_parameters(saved_model::String)
    return load_variational_autoencoder_parameters(FileIO.query(saved_model))
end

# end

