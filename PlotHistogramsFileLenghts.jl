import Plots, Distributions
import KernelDensity
using Measures

function plot_data(obs1_vecs::Dict{Int, Any}, 
    obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})

    file_lengths = zeros(Int,length(obs2_vecs))
    for file_id in keys(obs1_vecs)
        file_lengths[file_id] = length(obs1_vecs[file_id])
    end

    return Plots.histogram(file_lengths)
end

function load_data(save_dir::String)
    obs1_vecs = Dict{Int, Any}()
    obs2_vecs = Dict{Int, Any}()
    t_vecs = Dict{Int, Any}()

    function load_config(file_name::String)
        config = Dict{String, Any}()
        open(file_name, "r") do io
            for line in eachline(io)
                split_line = split(line, " = ")
                config[split_line[1]] = eval(Meta.parse(split_line[2]))
            end
        end
        return config
    end

    config = load_config(save_dir * "config.txt")
    n_samples = config["n_sample_files"]

    function load_observations(file_name::String)
        obs1_vec = Float64[]
        obs2_vec = Float64[]
        t_vec = Float64[]
        open(file_name, "r") do io
            for line in eachline(io)
                split_line = split(line)
                push!(obs1_vec, parse(Float64, split_line[1]))
                push!(obs2_vec, parse(Float64, split_line[2]))
                push!(t_vec, parse(Float64, split_line[3]))
            end
        end
        return obs1_vec, obs2_vec, t_vec
    end

    for sample_id in 1:n_samples
        obs1_vecs[sample_id], obs2_vecs[sample_id], t_vecs[sample_id] = 
            load_observations(save_dir * "observations_sample_$(sample_id).txt")
    end

    return obs1_vecs, obs2_vecs, t_vecs
end

function plot_data(save_dir::String)
    obs1_vecs, obs2_vecs, t_vecs = load_data(save_dir)
    plot = plot_data(obs1_vecs, obs2_vecs, t_vecs)
    Plots.plot!(title=save_dir)
    Plots.savefig(plot, save_dir * "file_sizes.pdf")
    return plot
end


plt = plot_data("saved_data/F_0.5/")
plt = plot_data("saved_data/F_0.55/")
plt = plot_data("saved_data/F_0.6/")
plt = plot_data("saved_data/F_0.65/")
plt = plot_data("saved_data/F_0.7/")
plt = plot_data("saved_data/F_0.75/")
plt = plot_data("saved_data/F_0.8/")
plt = plot_data("saved_data/F_0.85/")

