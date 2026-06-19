# module SurrogateProtocolVAE_HDF5

# using SurrogateProtocolVAE
import FileIO, HDF5

function make_enum_type_uint8(enum_dict::Dict)
    hdf5_enum_types = HDF5.Datatype(HDF5.API.h5t_create(HDF5.API.H5T_ENUM, sizeof(UInt8)))

    for (name, val) in enum_dict
        HDF5.API.h5t_enum_insert(hdf5_enum_types, name, Ref(val))
    end

    return hdf5_enum_types
end

function enum_to_dict(enum::DataType)
    return Dict(string(k) => k for k in instances(enum))
end

format_version::UInt64 = 1
format_version_minor::UInt64 = 0
quantumsavory_format_version::UInt64 = 1

@enum reference_state_types::UInt8 begin
    bell_pair = 0
end

@enum log_format_types::UInt8 begin
    fidelity            = 0
    state_vector        = 1
    werner_vector       = 2
    density_matrix      = 3
    pauli_observables   = 4
end

@enum simulation_mode::UInt8 begin
    repeated_single_shot = 0
    stateful = 1
end

reference_state_types_enum_dict = enum_to_dict(reference_state_types)
log_format_types_enum_dict = enum_to_dict(log_format_types)
simulation_mode_enum_dict = enum_to_dict(simulation_mode)
activation_function_enum_dict = enum_to_dict(ActivationFunction)

mandatory_qnet_group_attribute_keys = ["reference_state", "log_format", "simulation_mode"]
mandatory_metadata_group_attribute_keys = ["description", "simulator"]

function load_entanglement_consumer_log(file::FileIO.File{FileIO.DataFormat{:HDF5},String})
    function get_attribute_enum_types_dict(parent, attribute::String)
        dtype_id = HDF5.API.h5a_get_type(HDF5.attributes(parent)[attribute])  
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
        attr = HDF5.read_attribute(parent, attribute)
        return enum_dict[attr]
    end

    HDF5.h5open(file.filename, "r") do file
        qnet_group = HDF5.open_group(file, "qnet")
        metadata_group = HDF5.open_group(qnet_group, "metadata")
        simulation_log_group = HDF5.open_group(qnet_group, "simulation_log")

        format_version = HDF5.read_attribute(qnet_group, "format_version")
        format_version_minor = HDF5.read_attribute(qnet_group, "format_version_minor")
        log_format = read_attribute_enum(qnet_group, "log_format")
        reference_state = read_attribute_enum(qnet_group, "reference_state")

        simulation_time = HDF5.read_dataset(simulation_log_group, "time")
        simulation_state = HDF5.read_dataset(simulation_log_group, "state")

        if isa(simulation_state,Vector)
            simulation_state = collect(reshape(simulation_state,:,1)')
        end

        simulation_log = EntanglementConsumerLogSimulationLog(time = simulation_time, state = simulation_state)

        metadata_description = HDF5.read_attribute(metadata_group, "description")
        metadata_simulator = HDF5.read_attribute(metadata_group, "simulator")

        quantumsavory_metadata = Dict{String,Any}()
        if haskey(metadata_group, "quantumsavory")
            quantumsavory_group = HDF5.open_group(metadata_group, "quantumsavory")
            for attr in keys(HDF5.attributes(quantumsavory_group))
                quantumsavory_metadata[attr] = HDF5.read_attribute(quantumsavory_group, attr)
            end
        end

        metadata = EntanglementConsumerLogMetadata(
            description = metadata_description, 
            simulator = metadata_simulator, 
            quantumsavory_metadata = quantumsavory_metadata
        )

        return EntanglementConsumerLog(
            format_version = format_version,
            format_version_minor = format_version_minor,
            log_format = log_format,
            reference_state = reference_state,
            simulation_log = simulation_log,
            metadata = metadata
        )
    end
end

function load_variational_autoencoder_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String})
    activation_function_map = Dict{UInt8, ActivationFunction}(
        0 => _relu,
        1 => _gelu,
        2 => _tanh,
        3 => _sigmoid,
        4 => _softmax,
        5 => _leakyrelu,
        6 => _elu,
        7 => _swish,
        8 => _identity
    )

    HDF5.h5open(file.filename, "r") do file
        encoder_group = file["surrogate_qnet"]["autoencoder_parameters"]["encoder"]
        decoder_group = file["surrogate_qnet"]["autoencoder_parameters"]["decoder"]

        function read_group_layer_parameters(chain_group)
            layers = Vector{LayerParameters}(undef, length(keys(chain_group)))
            for i in eachindex(layers)
                layer_group = chain_group["layer $i"]
                new_layer = LayerParameters(
                    read(layer_group["node_bias"]),
                    read(layer_group["node_weights"]),
                    activation_function_map[UInt8(read(HDF5.attributes(layer_group)["activation_function"]))]
                )
                layers[i] = new_layer
            end

            return layers
        end

        encoder_layers = ChainParameters(read_group_layer_parameters(encoder_group))
        decoder_layers = ChainParameters(read_group_layer_parameters(decoder_group))
        return VariationalAutoencoderParameters(encoder_layers, decoder_layers)
    end
end

function FileIO.save(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, prot::SurrogateProtocolVAE; metadata::Dict{String,Any} = Dict{String,Any}())
    encoder_parameters = variational_autoencoder_parameters(prot)
    
    HDF5.h5open(file.filename, "w") do file
        reference_state = bell_pair
        log_format = pauli_observables
        simulation_mode = stateful

        surrogate_qnet_group = HDF5.create_group(file, "surrogate_qnet")
        metadata_group = HDF5.create_group(surrogate_qnet_group, "metadata")
        autoencoder_parameters_group = HDF5.create_group(surrogate_qnet_group, "autoencoder_parameters")
        encoder_group = HDF5.create_group(autoencoder_parameters_group, "encoder")
        decoder_group = HDF5.create_group(autoencoder_parameters_group, "decoder")
        quantumsavory_group = HDF5.create_group(metadata_group, "quantumsavory")
        user_optional_group = HDF5.create_group(metadata_group, "user_optional")

        ref_dt = make_enum_type_uint8(reference_state_types_enum_dict)
        log_dt = make_enum_type_uint8(log_format_types_enum_dict)
        sim_dt = make_enum_type_uint8(simulation_mode_enum_dict)
        lay_dt = make_enum_type_uint8(activation_function_enum_dict)
        scalar_space = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))

        attr_ref = HDF5.create_attribute(surrogate_qnet_group, "reference_state", ref_dt, scalar_space)
        attr_log = HDF5.create_attribute(surrogate_qnet_group, "log_format", log_dt, scalar_space)
        attr_sim = HDF5.create_attribute(surrogate_qnet_group, "simulation_mode", sim_dt, scalar_space)

        HDF5.write_attribute(attr_ref, ref_dt, UInt8(reference_state))
        HDF5.write_attribute(attr_log, log_dt, UInt8(log_format))
        HDF5.write_attribute(attr_sim, sim_dt, UInt8(simulation_mode))

        HDF5.write_attribute(surrogate_qnet_group, "format_version", format_version)
        HDF5.write_attribute(surrogate_qnet_group, "format_version_minor", format_version_minor)
        HDF5.write_attribute(metadata_group, "description", metadata["description"])
        HDF5.write_attribute(metadata_group, "simulator", metadata["simulator"])
        HDF5.write_attribute(quantumsavory_group, "quantumsavory_format_version", quantumsavory_format_version)

        for metadata_key in setdiff(keys(metadata), mandatory_qnet_group_attribute_keys, mandatory_metadata_group_attribute_keys)
            HDF5.write_attribute(user_optional_group, metadata_key, metadata[metadata_key])
        end

        function write_chain_to_file(chain, chain_group)
            for i in eachindex(chain.layers)
                layer = chain.layers[i]
                layer_group = HDF5.create_group(chain_group, "layer $i")
                attr_lay = HDF5.create_attribute(layer_group, "activation_function", lay_dt, scalar_space)
                HDF5.write_attribute(attr_lay, lay_dt, UInt8(layer.activation_function))
                HDF5.write(layer_group, "node_bias", layer.node_bias)
                HDF5.write(layer_group, "node_weights", layer.node_weights)
            end
        end

        write_chain_to_file(encoder_parameters.encoder, encoder_group)
        write_chain_to_file(encoder_parameters.decoder, decoder_group)
    end
end

# end