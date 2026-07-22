using ConcurrentSim, Plots, KernelDensity, Distributions, Measures
using LaTeXStrings

using QuantumSavory
using QuantumSavory.ProtocolZoo

using FileIO, HDF5, CSV, DataFrames
using GenerativeModelProtocols

function compare_kde_plot(vae_sim::Vector, ground_truth::Vector)
    vae_kde = kde(filter(!isnan, vae_sim))
    vae_kde = InterpKDE(vae_kde)
    vae_kde_pdf(x) = pdf(vae_kde,x)

    ground_truth_kde = kde(ground_truth)
    ground_truth_kde = InterpKDE(ground_truth_kde)
    ground_truth_kde_pdf(x) = pdf(ground_truth_kde,x)

    x0 = minimum(ground_truth)
    x1 = maximum(ground_truth)
    dx = (x1-x0) / 400.0
    plt_range = collect(x0:dx:x1)

    p = histogram(ground_truth, normalize=:pdf, label="SeQUeNCe simulated data")
    p = plot!(plt_range, vae_kde_pdf.(plt_range), label="VAE Surrogate (KDE from synthetic data)",frame=:box, linewidth=3)
    # plot!(plt_range, ground_truth_kde_pdf.(plt_range), label="SeQUeNCe Simulation",frame=:box, linewidth=3)
    
    return p
end

function compare_2d_kde_plot(Δt::Vector, obs::Vector, plot_title::String)
    kde = KernelDensity.kde((Δt, obs))
    kde = KernelDensity.InterpKDE(kde)
    kde_pdf(x,y) = Distributions.pdf(kde,x,y)

    x0 = minimum(Δt)
    x1 = maximum(Δt)
    dx = (x1 - x0) / 400.0

    y0 = minimum(obs)
    y1 = maximum(obs)
    dy = (y1 - y0) / 400.0

    plt = Plots.heatmap(x0:dx:x1, y0:dy:y1, kde_pdf, color=:viridis)
    Plots.plot!(frame=:box)
    Plots.plot!(xlabel = "Time to serve [s]")
    Plots.plot!(ylabel = "Fidelity")
    Plots.plot!(colorbar_title = "\np(Δt,F)")
    Plots.plot!(title = plot_title)
    return plt
end

function compare_latent_variables(protocol::GenerativeModelProtocol)
    function kde_interpolant(data::Vector)
        vae_kde = kde(data)
        vae_kde = InterpKDE(vae_kde)
        vae_kde_pdf(x) = pdf(vae_kde,x)

        return vae_kde_pdf
    end

    function plot_kde_interpolant(data::Vector, label::String)
        interp = kde_interpolant(data)
        x = collect(-10.0:0.01:10)
        return plot(x, interp.(x), label=label, linewidth=2)
    end

    function plot_kde_interpolant!(data::Vector, label::String)
        interp = kde_interpolant(data)
        x = collect(-10.0:0.01:10)
        return plot!(x, interp.(x), label=label, linewidth=2)
    end

    function reference_gaussian(x, μ, σ²)
        return exp(-((x - μ)^2)/(2.0 * σ²))/sqrt(2.0*π*σ²)
    end

    function plot_reference_gaussian!()
        x = collect(-10.0:0.01:10)
        return plot!(x, reference_gaussian.(x,0,1), style=:dash, label=L"\mathcal{N} (0,1)", linewidth=5)
    end

    z = GenerativeModelProtocols.encode(protocol.model, protocol.training_data)

    p = plot_kde_interpolant(z[1,:], "z₁")
    plot_kde_interpolant!(z[2,:], "z₂")
    plot_reference_gaussian!()

    plot!(p, frame=:box)
    plot!(xlabel="Latent Variable", ylabel="Probability Density")

    return p
end

function plot_train_loss(file_name::String)
    data = CSV.read(file_name, DataFrame)
    p = plot(data.loss, xlabel="Epochs", xaxis=:log, yaxis=:log, frame=:box, label="-ELBO")
    plot!(data.grad_norm, xlabel="Epochs", xaxis=:log, yaxis=:log, frame=:box, label="-ELBO norm")
    return p
end

base_dir = @__DIR__() * "/../../sequence_data/reference_data/"

for swap_prob in [0.8, 0.9]
    for num_latent_layers in 1:4
        save_dir = @__DIR__() * "/../../sequence_data/trained_surrogates/variational_autoencoder/latent_layers_$num_latent_layers/"
        
        model = GenerativeModelProtocols.VariationalAutoencoder(save_dir * "linear_17_swap_prob=$swap_prob.h5")
        sim_log = EntanglementConsumerLog(base_dir * "linear_17_swap_prob=$swap_prob.h5")
        training_data = hcat(sim_log.simulation_log.time, sim_log.simulation_log.state[:])
        training_data = collect(transpose(training_data))
        surrogate = GenerativeModelProtocol(model,training_data)

        num_vae_samples = 125000
        @time sim_data = surrogate(num_vae_samples)

        p1 = compare_kde_plot(sim_data[:,1][:], surrogate.training_data[1,:][:])
        plot!(p1, ylabel="Probability\nDensity")
        plot!(p1, xlabel="Time to serve [ms]")

        p2 = compare_kde_plot(sim_data[:,2][:], surrogate.training_data[2,:][:])
        plot!(p2, ylabel="Probability\nDensity")
        plot!(p2, xlabel="Fidelity")
        plot!(p2, legend=false)

        p = plot(p1,p2,layout=(2,1))
        savefig(p, save_dir * "linear_17_swap_prob=$swap_prob.png")
        savefig(p, save_dir * "linear_17_swap_prob=$swap_prob.pdf")
        savefig(p, save_dir * "linear_17_swap_prob=$swap_prob.svg")

        p1 = compare_2d_kde_plot(surrogate.training_data[1,:][:], surrogate.training_data[2,:][:], "2D KDE from SeQUeNCe Data")
        p2 = compare_2d_kde_plot(sim_data[:,1][:], sim_data[:,2][:], "2D KDE from VAE Surrogate")

        plot!(p1,xlims=(12,80),ylims=(0.43,0.48))
        plot!(p2,xlims=(12,80),ylims=(0.43,0.48))

        p = plot(p1,p2,layout=(2,1), right_margin = 5mm)
        savefig(p, save_dir * "2d_kde_linear_17_swap_prob=$swap_prob.png")
        savefig(p, save_dir * "2d_kde_linear_17_swap_prob=$swap_prob.pdf")
        savefig(p, save_dir * "2d_kde_linear_17_swap_prob=$swap_prob.svg")

        p = compare_latent_variables(surrogate)
        savefig(p, save_dir * "latent_comp_linear_17_swap_prob=$swap_prob.png")
        savefig(p, save_dir * "latent_comp_linear_17_swap_prob=$swap_prob.pdf")
        savefig(p, save_dir * "latent_comp_linear_17_swap_prob=$swap_prob.svg")

        p = plot_train_loss(save_dir * "linear_17_swap_prob=$swap_prob.csv")
        savefig(p, save_dir * "ELBO_training_swap_prob=$swap_prob.png")
        savefig(p, save_dir * "ELBO_training_swap_prob=$swap_prob.pdf")
        savefig(p, save_dir * "ELBO_training_swap_prob=$swap_prob.svg")
    end
end

for swap_prob in [0.8, 0.9]
    save_dir = @__DIR__() * "/../../sequence_data/trained_surrogates/diffusion_model/"
    
    model = GenerativeModelProtocols.DiffusionModel(save_dir * "linear_17_swap_prob=$swap_prob.h5")
    sim_log = EntanglementConsumerLog(base_dir * "linear_17_swap_prob=$swap_prob.h5")
    training_data = hcat(sim_log.simulation_log.time, sim_log.simulation_log.state[:])
    training_data = collect(transpose(training_data))
    surrogate = GenerativeModelProtocol(model,training_data)

    num_vae_samples = 125000
    @time sim_data = surrogate(num_vae_samples)

    p1 = compare_kde_plot(sim_data[:,1][:], surrogate.training_data[1,:][:])
    plot!(p1, ylabel="Probability\nDensity")
    plot!(p1, xlabel="Time to serve [ms]")

    p2 = compare_kde_plot(sim_data[:,2][:], surrogate.training_data[2,:][:])
    plot!(p2, ylabel="Probability\nDensity")
    plot!(p2, xlabel="Fidelity")
    plot!(p2, legend=false)

    p = plot(p1,p2,layout=(2,1))
    savefig(p, save_dir * "linear_17_swap_prob=$swap_prob.png")
    savefig(p, save_dir * "linear_17_swap_prob=$swap_prob.pdf")
    savefig(p, save_dir * "linear_17_swap_prob=$swap_prob.svg")

    p1 = compare_2d_kde_plot(surrogate.training_data[1,:][:], surrogate.training_data[2,:][:], "2D KDE from SeQUeNCe Data")
    p2 = compare_2d_kde_plot(sim_data[:,1][:], sim_data[:,2][:], "2D KDE from VAE Surrogate")

    # plot!(p1,xlims=(12,80),ylims=(0.43,0.48))
    # plot!(p2,xlims=(12,80),ylims=(0.43,0.48))

    p = plot(p1,p2,layout=(2,1), right_margin = 5mm)
    savefig(p, save_dir * "2d_kde_linear_17_swap_prob=$swap_prob.png")
    savefig(p, save_dir * "2d_kde_linear_17_swap_prob=$swap_prob.pdf")
    savefig(p, save_dir * "2d_kde_linear_17_swap_prob=$swap_prob.svg")

    p = plot_train_loss(save_dir * "linear_17_swap_prob=$swap_prob.csv")
    savefig(p, save_dir * "ELBO_training_swap_prob=$swap_prob.png")
    savefig(p, save_dir * "ELBO_training_swap_prob=$swap_prob.pdf")
    savefig(p, save_dir * "ELBO_training_swap_prob=$swap_prob.svg")
end

