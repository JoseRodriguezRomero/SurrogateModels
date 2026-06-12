import Plots, Distributions
import KernelDensity, StatsBase
import KernelDensity: kernel_dist

using Base.Threads
using Measures, DataInterpolations
using LaTeXStrings, Interpolations

include("load_log_files.jl")

function plot_data_1d(obs1_vecs::Dict{Int, Any}, obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})
    obs1_vec = zeros(Float64,0)
    obs2_vec = zeros(Float64,0)
    t_diff_vec = zeros(Float64,0)
    for file_id in keys(obs1_vecs)
        obs1_vec = vcat(obs1_vec, obs1_vecs[file_id][1:(end-1)])
        obs2_vec = vcat(obs2_vec, obs2_vecs[file_id][1:(end-1)])
        t_diff_vec = vcat(t_diff_vec, diff(t_vecs[file_id]))
    end

    # obs1_vec = vcat(obs1_vec, -obs1_vec)
    # obs2_vec = vcat(obs2_vec, -obs2_vec)
    # t_diff_vec = vcat(t_diff_vec, -t_diff_vec)

    p1 = Plots.histogram(obs2_vec, normalize=:pdf, bins=100,
        label="QSavory Data")
    kde1 = KernelDensity.kde(obs2_vec)
    kde1 = KernelDensity.InterpKDE(kde1)
    kde_pdf_1(x) = Distributions.pdf(kde1,x)

    dx = (maximum(obs2_vec) - minimum(obs2_vec)) / 100.0
    plt_x = minimum(obs2_vec):dx:maximum(obs2_vec)
    plt_y = kde_pdf_1.(plt_x)
    Plots.plot!(plt_x, plt_y, lw = 2, label = "Kernel Density Estimation")
    
    Plots.plot!(xlabel = "(O₁) Fidelity of consumed pairs", ylabel = "p₁(O₁)")
    Plots.plot!(frame = :box)

    p2 = Plots.histogram(t_diff_vec, normalize=:pdf, bins=100, label="QSavory Data")
    kde2 = KernelDensity.kde(t_diff_vec)
    kde2 = KernelDensity.InterpKDE(kde2)
    kde_pdf_2(x) = Distributions.pdf(kde2,x)

    dx = (maximum(t_diff_vec) - minimum(t_diff_vec)) / 600.0
    plt_x = minimum(t_diff_vec):dx:maximum(t_diff_vec)
    plt_y = kde_pdf_2.(plt_x)
    Plots.plot!(plt_x, plt_y, lw = 2, label = "Kernel Density Estimation")

    Plots.plot!(xlabel = "(Δt) Time between consumptions [s]", ylabel = "pₜ(Δt)")
    Plots.plot!(frame = :box)

    return Plots.plot(p1, p2, layout = (2,1), 
        size=(800,400), left_margin = 3mm, bottom_margin = 4mm)
end

function plot_corr_2d(obs1_vecs::Dict{Int, Any}, obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})
    obs1_vec = zeros(Float64,0)
    obs2_vec = zeros(Float64,0)
    t_diff_vec = zeros(Float64,0)
    for file_id in keys(obs1_vecs)
        obs1_vec = vcat(obs1_vec, obs1_vecs[file_id][1:(end-1)])
        obs2_vec = vcat(obs2_vec, obs2_vecs[file_id][1:(end-1)])
        t_diff_vec = vcat(t_diff_vec, diff(t_vecs[file_id]))
    end

    kde1 = KernelDensity.kde(obs2_vec)
    kde1 = KernelDensity.InterpKDE(kde1)
    itp1(x) = Distributions.pdf(kde1,x)

    kde2 = KernelDensity.kde(t_diff_vec)
    kde2 = KernelDensity.InterpKDE(kde2)
    itp2(x) = Distributions.pdf(kde2,x)

    kde12 = KernelDensity.kde((obs2_vec, t_diff_vec))
    kde12 = KernelDensity.InterpKDE(kde12)
    itp12(x,y) = Distributions.pdf(kde12,x,y)

    sample_inds = 1:length(obs2_vec)
    sample_inds = StatsBase.sample(sample_inds, min(400, length(sample_inds)), replace=false)
    obs2_vec = obs2_vec[sample_inds]
    t_diff_vec = t_diff_vec[sample_inds]

    corr_data = itp12.(obs2_vec[:], t_diff_vec[:])
    uncorr_data = itp1.(obs2_vec[:]) .* itp2.(t_diff_vec[:])

    dx = (maximum(uncorr_data) - minimum(uncorr_data)) / 10.0
    model_x = collect(minimum(uncorr_data):dx:maximum(uncorr_data))
    model_y = model_x

    plt = Plots.scatter(uncorr_data, corr_data, label = "Simulated data")
    Plots.plot!(model_x, model_y, label = "f(x) = x", lw = 4)
    Plots.plot!(xlabel = "pₜ(Δt) p₁(O₁)")
    Plots.plot!(ylabel = "pₜ₁(Δt,O₁)")
    Plots.plot!(frame=:box)
    return plt
end

function plot_data_2d(obs1_vecs::Dict{Int, Any}, obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})
    obs1_vec = zeros(Float64,0)
    obs2_vec = zeros(Float64,0)
    t_diff_vec = zeros(Float64,0)
    for file_id in keys(obs1_vecs)
        obs1_vec = vcat(obs1_vec, obs1_vecs[file_id][1:(end-1)])
        obs2_vec = vcat(obs2_vec, obs2_vecs[file_id][1:(end-1)])
        t_diff_vec = vcat(t_diff_vec, diff(t_vecs[file_id]))
    end

    kde = KernelDensity.kde((obs2_vec, t_diff_vec))
    kde = KernelDensity.InterpKDE(kde)
    kde_pdf(x,y) = Distributions.pdf(kde,x,y)

    x0 = minimum(obs2_vec)
    x1 = maximum(obs2_vec)
    dx = (x1 - x0) / 400.0

    y0 = minimum(t_diff_vec)
    y1 = maximum(t_diff_vec)
    dy = (y1 - y0) / 400.0

    plt = Plots.heatmap(x0:dx:x1, y0:dy:y1, kde_pdf, color=:viridis)
    Plots.plot!(frame=:box)
    Plots.plot!(xlabel = "O₁")
    Plots.plot!(ylabel = "Δt")
    Plots.plot!(colorbar_title = "pₜ₁(Δt,O₁)")
    Plots.plot!(title = "Simulated data in QSavory\n(2D Kernel Density Estimation)")
    return plt
end

function plot_data_model_2d(obs1_vecs::Dict{Int, Any}, obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})
    obs1_vec = zeros(Float64,0)
    obs2_vec = zeros(Float64,0)
    t_diff_vec = zeros(Float64,0)
    for file_id in keys(obs1_vecs)
        obs1_vec = vcat(obs1_vec, obs1_vecs[file_id][1:(end-1)])
        obs2_vec = vcat(obs2_vec, obs2_vecs[file_id][1:(end-1)])
        t_diff_vec = vcat(t_diff_vec, diff(t_vecs[file_id]))
    end

    kde1 = KernelDensity.kde(obs2_vec)
    kde1 = KernelDensity.InterpKDE(kde1)
    kde2 = KernelDensity.kde(t_diff_vec)
    kde2 = KernelDensity.InterpKDE(kde2)

    itp1(x) = Distributions.pdf(kde1,x)
    itp2(x) = Distributions.pdf(kde2,x)

    dx = (maximum(obs2_vec) - minimum(obs2_vec)) / 400.0
    x_pts = minimum(obs2_vec):dx:maximum(obs2_vec)

    dy = (maximum(t_diff_vec) - minimum(t_diff_vec)) / 400.0
    y_pts = minimum(t_diff_vec):dy:maximum(t_diff_vec)

    p_model(O₁,Δt) =  itp2(Δt) * itp1(O₁)
    plt = Plots.heatmap(x_pts, y_pts, p_model, color=:viridis)

    Plots.plot!(frame=:box)
    Plots.plot!(xlabel = "O₁")
    Plots.plot!(ylabel = "Δt")
    Plots.plot!(colorbar_title = "pₜ(Δt) p₁(O₁)")
    Plots.plot!(title = "Treating O₁ and Δt as\nindependent random variables")
    return plt
end

function vae_model(obs1_vecs::Dict{Int, Any}, obs2_vecs::Dict{Int, Any}, t_vecs::Dict{Int, Any})
    obs1_vec = zeros(Float64,0)
    obs2_vec = zeros(Float64,0)
    t_diff_vec = zeros(Float64,0)
    for file_id in keys(obs1_vecs)
        obs1_vec = vcat(obs1_vec, obs1_vecs[file_id][1:(end-1)])
        obs2_vec = vcat(obs2_vec, obs2_vecs[file_id][1:(end-1)])
        t_diff_vec = vcat(t_diff_vec, diff(t_vecs[file_id]))
    end

    
end

function load_data(save_dir::String)
    obs1_vecs = Dict{Int, Any}()
    obs2_vecs = Dict{Int, Any}()
    t_vecs = Dict{Int, Any}()

    file_names = filter(x -> endswith(x, ".h5"), readdir(save_dir, join=true))
    for sample_id in eachindex(file_names)
        try
            consumer_log = load_entanglement_consumer_log(file_names[sample_id])

            t_vecs[sample_id] = consumer_log.simulation_log.time
            obs1_vecs[sample_id] = consumer_log.simulation_log.state[1,:]
            obs2_vecs[sample_id] = consumer_log.simulation_log.state[2,:]
        catch
            display("File failed: $(file_names[sample_id])")
        end
    end

    return obs1_vecs, obs2_vecs, t_vecs
end

function plot_data(save_dir::String)
    obs1_vecs, obs2_vecs, t_vecs = load_data(save_dir)
    plot = plot_data_1d(obs1_vecs, obs2_vecs, t_vecs)
    Plots.savefig(plot, save_dir * "histograms_univariate.pdf")
    Plots.savefig(plot, save_dir * "histograms_univariate.svg")

    plot1 = plot_data_2d(obs1_vecs, obs2_vecs, t_vecs)
    plot2 = plot_data_model_2d(obs1_vecs, obs2_vecs, t_vecs)
    plot = Plots.plot(plot1, plot2, layout=(1,2), size=(900,300), left_margin = 4mm, bottom_margin = 4mm, top_margin = 8mm)
    Plots.savefig(plot, save_dir * "histograms_bivariate.pdf")
    Plots.savefig(plot, save_dir * "histograms_bivariate.svg")

    plot = plot_corr_2d(obs1_vecs, obs2_vecs, t_vecs)
    Plots.savefig(plot, save_dir * "model_corr_plot.pdf")
    Plots.savefig(plot, save_dir * "model_corr_plot.svg")

    return plot
end

for F in 0.5:0.05:0.95
    plot_data("saved_data/F_$(F)/")
    println("Finished plotting for F = $(F)")
end

