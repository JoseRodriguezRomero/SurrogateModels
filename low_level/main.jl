import Plots, Distributions
import KernelDensity
using Measures

function plot_data(obs1_vecs::Dict{Int, Any}, 
    obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})

    obs1_vec = zeros(Float64,0)
    obs2_vec = zeros(Float64,0)
    t_diff_vec = zeros(Float64,0)
    for thread_id in keys(obs1_vecs)
        obs1_vec = vcat(obs1_vec, obs1_vecs[thread_id][1:(end-1)])
        obs2_vec = vcat(obs2_vec, obs2_vecs[thread_id][1:(end-1)])
        t_diff_vec = vcat(t_diff_vec, diff(t_vecs[thread_id]))

        # i0 = round(Int, length(obs1_vecs[thread_id]) / 90)
        # obs1_vec = obs1_vec[i0:end]
        # obs2_vec = obs2_vec[i0:end]
        # t_diff_vec = t_diff_vec[i0:end]
    end

    return Plots.histogram2d(obs2_vec, t_diff_vec, bins=(250,250), 
        normalize=:pdf, show_empty_bins=false, color=:jet)

    # bandwidth = (maximum(obs2_vec) - minimum(obs2_vec)) / 100.0
    # k = KernelDensity.kde(obs2_vec, bandwidth = bandwidth)
    # p1 = Plots.histogram(obs2_vec, normalize=:pdf, bins=200, 
    #     label="QuantumSavory")
    # Plots.plot!(k.x, k.density, lw = 4, label="KDE")
    # Plots.plot!(xlabel = "Fidelity of consumed pairs", 
    #     ylabel = "Probability\nDensity")
    # Plots.plot!(frame = :box)
    # # Plots.plot!(xlims=(0.0,0.55))
    # # Plots.plot!(ylims=(0.0,40.0))

    # bandwidth = (maximum(t_diff_vec) - minimum(t_diff_vec)) / 100.0
    # k = KernelDensity.kde(t_diff_vec, bandwidth = bandwidth)
    # p2 = Plots.histogram(t_diff_vec, normalize=:pdf, bins=200, 
    #     label="QuantumSavory")
    # Plots.plot!(k.x, k.density, lw = 4, label="KDE")
    # Plots.plot!(xlabel = "Time between consumptions (s)", 
    #     ylabel = "Probability\nDensity")
    # Plots.plot!(frame = :box)
    # # Plots.plot!(xlims=(0.0,15.0))
    # # Plots.plot!(ylims=(0.0,1.0))

    # return Plots.plot(p1, p2, layout = (2,1), 
    #     size=(800,400), left_margin = 4mm, bottom_margin = 4mm)
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
    Plots.savefig(plot, save_dir * "histograms.pdf")
    return plot
end


plot_data("low_level/saved_data/F_0.5/")

