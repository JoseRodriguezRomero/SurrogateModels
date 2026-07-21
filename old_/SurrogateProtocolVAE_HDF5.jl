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
format_version_minor::UInt64 = 1
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

@enum simulation_mode_types::UInt8 begin
    repeated_single_shot = 0
    stateful = 1
end

mandatory_qnet_group_attribute_keys = ["reference_state", "log_format", "simulation_mode"]
mandatory_metadata_group_attribute_keys = ["description", "simulator"]

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
        encoder_group = file["surrogate_qnet"]["autoencoder_parameters"]["encoders"]
        decoder_group = file["surrogate_qnet"]["autoencoder_parameters"]["decoders"]

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

        encoders = Vector{ChainParameters}(undef, length(keys(encoder_group)))
        decoders = Vector{ChainParameters}(undef, length(keys(decoder_group)))

        for i in eachindex(encoders)
            encoder_chain_group = encoder_group["encoder $i"]
            encoders[i] = ChainParameters(read_group_layer_parameters(encoder_chain_group))
        end

        for i in eachindex(decoders)
            decoder_chain_group = decoder_group["decoder $i"]
            decoders[i] = ChainParameters(read_group_layer_parameters(decoder_chain_group))
        end

        return VariationalAutoencoderParameters(encoders, decoders)
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
        encoders_group = HDF5.create_group(autoencoder_parameters_group, "encoders")
        decoders_group = HDF5.create_group(autoencoder_parameters_group, "decoders")
        quantumsavory_group = HDF5.create_group(metadata_group, "quantumsavory")
        user_optional_group = HDF5.create_group(metadata_group, "user_optional")

        ref_dt = make_enum_type_uint8(enum_to_dict(reference_state_types))
        log_dt = make_enum_type_uint8(enum_to_dict(log_format_types))
        sim_dt = make_enum_type_uint8(enum_to_dict(simulation_mode_types))
        lay_dt = make_enum_type_uint8(enum_to_dict(ActivationFunction))
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

        function write_encoder_to_file(encoder::Vector)
            for i in eachindex(encoder)
                encoder_group = HDF5.create_group(encoders_group, "encoder $i")
                write_chain_to_file(encoder[i], encoder_group)
            end
        end

        function write_decoder_to_file(decoder::Vector)
            for i in eachindex(decoder)
                decoder_group = HDF5.create_group(decoders_group, "decoder $i")
                write_chain_to_file(decoder[i], decoder_group)
            end
        end

        write_encoder_to_file(encoder_parameters.encoders)
        write_decoder_to_file(encoder_parameters.decoders)
    end
end

# end