# module SurrogateProtocolVAE_HDF5

# using SurrogateProtocolVAE
import FileIO, HDF5

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