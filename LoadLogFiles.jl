using HDF5, CSV, DataFrames, FileIO

@kwdef struct EntanglementConsumerLog
    t::Vector{Float64} = zeros(Float64,0)
    obs1::Vector{Float64} = zeros(Float64,0)
    obs2::Vector{Float64} = zeros(Float64,0)
    metadata::Union{Dict{String, Any}, Nothing} = nothing
end

function load_entanglement_consumer_log(save_file::FileIO.File{FileIO.DataFormat{:CSV},String})
    df = CSV.read(save_file.filename, DataFrame)
    return EntanglementConsumerLog(t = df.t, obs1 = df.obs1, obs2 = df.obs2)
end

function load_entanglement_consumer_log(save_file::FileIO.File{FileIO.DataFormat{:HDF5},String})
    log_t = zeros(Float64,0)
    log_obs1 = zeros(Float64,0)
    log_obs2 = zeros(Float64,0)
    log_metadata = nothing

    h5open(save_file.filename, "r") do file
        log_t = read(file["t"])
        log_obs1 = read(file["obs1"])
        log_obs2 = read(file["obs2"])

        if haskey(file, "metadata_keys")
            log_metadata = Dict{String, Any}()
            metadata_keys = read(file["metadata_keys"])
            metadata_values = read(file["metadata_values"])
            for i in eachindex(metadata_keys)
                key = metadata_keys[i]
                value = metadata_values[i]
                log_metadata[key] = value
            end
        end
    end

    return EntanglementConsumerLog(log_t, log_obs1, log_obs2, log_metadata)
end

function load_entanglement_consumer_log(file_name::String)
    return load_entanglement_consumer_log(query(file_name))
end

