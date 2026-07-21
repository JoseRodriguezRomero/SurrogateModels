using ConcurrentSim, Plots, KernelDensity, Distributions, Measures
using LaTeXStrings

using QuantumSavory
using QuantumSavory.ProtocolZoo

using FileIO, HDF5, CSV, DataFrames
using GenerativeModelProtocols

function compare_kde_plot(vae_sim::Vector, ground_truth::Vector)
    vae_kde = kde(vae_sim)
    vae_kde = InterpKDE(vae_kde)
    vae_kde_pdf(x) = pdf(vae_kde,x)

    ground_truth_kde = kde(ground_truth)
    ground_truth_kde = InterpKDE(ground_truth_kde)
    ground_truth_kde_pdf(x) = pdf(ground_truth_kde,x)

    x0 = minimum(ground_truth)
    x1 = maximum(ground_truth)
    dx = (x1-x0) / 400.0
    plt_range = collect(x0:dx:x1)

    p = histogram(ground_truth, normalize=:pdf, label="QuantumSavory simulated data")
    plot!(plt_range, vae_kde_pdf.(plt_range), label="VAE Surrogate (KDE from synthetic data)",frame=:box, linewidth=3)
    # plot!(plt_range, ground_truth_kde_pdf.(plt_range), label="QSavory Simulation",frame=:box, linewidth=3)
    
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
    Plots.plot!(ylabel = "<X⊗X>")
    Plots.plot!(colorbar_title = "\np(Δt,<X⊗X>)")
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
    plot_kde_interpolant!(z[3,:], "z₃")
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

function read_data(file_path::String)
    sim_log = EntanglementConsumerLog(file_path)
    training_data = hcat(
        sim_log.simulation_log.time, 
        sim_log.simulation_log.state[1,:], 
        sim_log.simulation_log.state[2,:]
    )
    training_data = hcat(
        reshape(diff(training_data[:,1]), :, 1),
        training_data[2:end,2:end]
    )

    return collect(transpose(training_data))
end

function read_all_data(base_dir::String)
    files = filter(f -> endswith(f, ".h5"), readdir(base_dir))
    training_data = zeros(Float64,3,0)

    for file in files
        training_data = hcat(training_data, read_data(base_dir * file))
    end

    return training_data
end

base_dir = @__DIR__() * "/../../qsavory_data/reference_data/"

for fidelity in [0.75, 0.95]
    for num_latent_layers in 1:4
        save_dir = @__DIR__() * "/../../qsavory_data/trained_surrogates/variational_autoencoder/latent_layers_$num_latent_layers/F_$fidelity/"

        model = GenerativeModelProtocols.VariationalAutoencoder(save_dir * "F_$fidelity.h5")
        training_data = read_all_data(@__DIR__() * "/../../qsavory_data/reference_data/F_$fidelity/")
        surrogate = GenerativeModelProtocol(model,training_data)

        num_vae_samples = 125000
        @time sim_data = surrogate(num_vae_samples)

        display(size(training_data))
        display(size(sim_data))

        p1 = compare_kde_plot(sim_data[:,1], surrogate.training_data[1,:])
        plot!(p1, ylabel="Probability\nDensity")
        plot!(p1, xlabel="Time to serve [s]")
        plot!(xlims=[-1,15])

        p2 = compare_kde_plot(sim_data[:,3], surrogate.training_data[3,:])
        plot!(p2, ylabel="Probability\nDensity")
        plot!(p2, xlabel="<X⊗X>")
        plot!(p2, legend=false)

        p = plot(p1,p2,layout=(2,1))
        savefig(p, save_dir * "F_$fidelity.png")
        savefig(p, save_dir * "F_$fidelity.svg")
        savefig(p, save_dir * "F_$fidelity.pdf")

        p1 = compare_2d_kde_plot(surrogate.training_data[1,:], surrogate.training_data[3,:], "2D KDE from QuantumSavory Data")
        p2 = compare_2d_kde_plot(sim_data[:,1], sim_data[:,3], "2D KDE from VAE Surrogate")

        if fidelity == 0.75
            plot!(p1,xlims=(0,10),ylims=(0.01,0.06))
            plot!(p2,xlims=(0,10),ylims=(0.01,0.06))
        elseif fidelity == 0.95
            plot!(p1,xlims=(0,10),ylims=(0.1,0.5))
            plot!(p2,xlims=(0,10),ylims=(0.1,0.5))
        end

        p = plot(p1,p2,layout=(2,1), right_margin = 5mm)
        savefig(p, save_dir * "2d_kde_F_$fidelity.png")
        savefig(p, save_dir * "2d_kde_F_$fidelity.svg")
        savefig(p, save_dir * "2d_kde_F_$fidelity.pdf")

        p = compare_latent_variables(surrogate)
        savefig(p, save_dir * "latent_comp_F_$fidelity.png")
        savefig(p, save_dir * "latent_comp_F_$fidelity.svg")
        savefig(p, save_dir * "latent_comp_F_$fidelity.pdf")

        p = plot_train_loss(save_dir * "F_$fidelity.csv")
        savefig(p, save_dir * "ELBO_training.pdf")
        savefig(p, save_dir * "ELBO_training.svg")
        savefig(p, save_dir * "ELBO_training.png")
    end
end

