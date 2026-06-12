using HDF5, CSV, DataFrames, FileIO

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

function load_entanglement_consumer_log(save_file::FileIO.File{FileIO.DataFormat{:CSV},String})
    df = CSV.read(save_file.filename, DataFrame)
    consumer_log = EntanglementConsumerLog()
    consumer_log.simulation_log.time = df.t
    consumer_log.simulation_log.state = hcat(df.obs1,df.obs2)

    return consumer_log
end

function load_entanglement_consumer_log(save_file::FileIO.File{FileIO.DataFormat{:HDF5},String})
    consumer_log = nothing

    function get_attribute_enum_types_dict(parent, attribute::String)
        dtype_id = HDF5.API.h5a_get_type(attributes(parent)[attribute])  
        dtype = HDF5.Datatype(dtype_id)

        nmembers = HDF5.API.h5t_get_nmembers(dtype)
        type_size = HDF5.API.h5t_get_size(dtype)
        
        T = if type_size == 1; Int8
            elseif type_size == 2; Int16
            elseif type_size == 4; Int32
            else; Int64
        end

        enum_dict = Dict{T, String}()
        
        for i in 0:(nmembers - 1)
            name = HDF5.API.h5t_get_member_name(dtype, i)
            
            val_buf = zeros(UInt8, type_size)
            
            ccall(
                (:H5Tget_member_value, HDF5.API.libhdf5), 
                Cint, 
                (HDF5.API.hid_t, Cuint, Ptr{Cvoid}), 
                dtype, i, val_buf
            )
            
            raw_val = reinterpret(T, val_buf)[1]
            
            is_big_endian = (HDF5.API.h5t_get_order(dtype) == HDF5.API.H5T_ORDER_BE)
            actual_val = is_big_endian ? ntoh(raw_val) : raw_val
            
            enum_dict[actual_val] = name
        end

        close(dtype)
        
        return enum_dict
    end

    function read_attribute_enum(parent, attribute::String)
        enum_dict = get_attribute_enum_types_dict(parent, attribute)
        attr = read_attribute(parent, attribute)
        return enum_dict[attr]
    end

    h5open(save_file.filename, "r") do file
        qnet_group = open_group(file, "qnet")
        metadata_group = open_group(qnet_group, "metadata")
        simulation_log_group = open_group(qnet_group, "simulation_log")
        quantumsavory_group = open_group(metadata_group, "quantumsavory")

        format_version = read_attribute(qnet_group, "format_version")
        format_version_minor = read_attribute(qnet_group, "format_version_minor")
        log_format = read_attribute_enum(qnet_group, "log_format")
        reference_state = read_attribute_enum(qnet_group, "reference_state")

        simulation_time = read_dataset(simulation_log_group, "time")
        simulation_state = read_dataset(simulation_log_group, "state")
        simulation_log = EntanglementConsumerLogSimulationLog(time = simulation_time, state = simulation_state)

        metadata_description = read_attribute(metadata_group, "description")
        metadata_simulator = read_attribute(metadata_group, "simulator")

        quantumsavory_metadata = Dict{String,Any}()
        for attr in keys(attributes(quantumsavory_group))
            quantumsavory_metadata[attr] = read_attribute(quantumsavory_group, attr)
        end

        metadata = EntanglementConsumerLogMetadata(description = metadata_description, simulator = metadata_simulator, quantumsavory_metadata = quantumsavory_metadata)

        consumer_log = EntanglementConsumerLog(
            format_version = format_version,
            format_version_minor = format_version_minor,
            log_format = log_format,
            reference_state = reference_state,
            simulation_log = simulation_log,
            metadata = metadata
        )
    end

    return consumer_log
end

function load_entanglement_consumer_log(file_name::String)
    return load_entanglement_consumer_log(FileIO.query(file_name))
end

function test_open_log(test_file_name::String = "saved_data/F_0.5/sample_1.h5")
    if isfile(test_file_name)
        return load_entanglement_consumer_log(test_file_name)
    else
        @error "No saved data to load! Run some simulations."
    end
end

