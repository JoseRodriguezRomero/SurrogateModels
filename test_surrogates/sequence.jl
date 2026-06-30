using ConcurrentSim, Plots, KernelDensity, Distributions, Measures

include("../SurrogateProtocolVAE.jl")

using .SurrogateProtocolVAE

base_dir = @__DIR__() * "/"
swap_prob = 0.9

sim = Simulation()
surrogate = SurrogateProtocolVAE.SurrogateProtocolVAE(sim;
    single_shot = true,
    training_data_path = base_dir *  "../sequence_data/linear_17_swap_prob=$swap_prob.h5", 
    saved_model = base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.h5"
)

num_vae_samples = 125000
@time sim_data = surrogate(num_vae_samples)

function compare_kde_plot(vae_sim::Vector, ground_truth::Vector)
    vae_kde = kde(vae_sim)
    vae_kde = InterpKDE(vae_kde)
    vae_kde_pdf(x) = pdf(vae_kde,x)

    ground_truth_kde = kde(ground_truth)
    ground_truth_kde = InterpKDE(ground_truth_kde)
    ground_truth_kde_pdf(x) = pdf(ground_truth_kde,x)

    x0 = minimum(ground_truth)
    x1 = maximum(ground_truth)
    dx = (x1-x0) / 300.0
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

function compare_latent_variables(protocol::SurrogateProtocolVAE.SurrogateProtocolVAE)
    encoder = protocol.encoder_chain
    data = protocol.training_data
    
    enc_out = [encoder(data[:,i]) for i in 1:size(data,2)]
    enc_out = collect(transpose(reduce(hcat,enc_out)))

    function reparametrization_trick(μ, logσ²)
        ϵ = randn(Float64, size(μ))
        return μ .+ exp(logσ² / 2.0) .* ϵ 
    end

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

    num_latent = round(Int, size(enc_out,2) / 2.0)
    μ_all = enc_out[:,1:num_latent]
    logσ²_all = enc_out[:,(num_latent+1):end]

    μ_all = repeat(μ_all,1,100)
    logσ²_all = repeat(logσ²_all,1,100)

    z₁ = reparametrization_trick.(μ_all[:,1], logσ²_all[:,1])
    z₂ = reparametrization_trick.(μ_all[:,2], logσ²_all[:,2])

    p = plot_kde_interpolant(z₁, "z₁")
    plot_kde_interpolant!(z₂, "z₂")
    plot_reference_gaussian!()

    plot!(p, frame=:box)
    plot!(xlabel="Latent Variable", ylabel="Probability Density")

    return p
end

simulation_window = surrogate.simulation_window_size
p1_range = (0*simulation_window+1):(1*simulation_window)
p2_range = (1*simulation_window+1):(2*simulation_window)

p1 = compare_kde_plot(sim_data[:,p1_range][:], surrogate.training_data[p1_range,:][:])
plot!(p1, ylabel="Probability\nDensity")
plot!(p1, xlabel="Time to serve [ms]")

p2 = compare_kde_plot(sim_data[:,p2_range][:], surrogate.training_data[p2_range,:][:])
plot!(p2, ylabel="Probability\nDensity")
plot!(p2, xlabel="Fidelity")
plot!(p2, legend=false)

p = plot(p1,p2,layout=(2,1))
savefig(p, base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.png")
savefig(p, base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.pdf")
savefig(p, base_dir * "../trained_surrogates/sequence/linear_17_swap_prob=$swap_prob.svg")
display(p)

p1 = compare_2d_kde_plot(surrogate.training_data[p1_range,:][:], surrogate.training_data[p2_range,:][:], "2D KDE from SeQUeNCe Data")
p2 = compare_2d_kde_plot(sim_data[:,p1_range][:], sim_data[:,p2_range][:], "2D KDE from VAE Surrogate")

plot!(p1,xlims=(12,80),ylims=(0.43,0.48))
plot!(p2,xlims=(12,80),ylims=(0.43,0.48))

p = plot(p1,p2,layout=(2,1), right_margin = 5mm)
savefig(p, base_dir * "../trained_surrogates/sequence/2d_kde_linear_17_swap_prob=$swap_prob.png")
savefig(p, base_dir * "../trained_surrogates/sequence/2d_kde_linear_17_swap_prob=$swap_prob.pdf")
savefig(p, base_dir * "../trained_surrogates/sequence/2d_kde_linear_17_swap_prob=$swap_prob.svg")
display(p)

p = compare_latent_variables(surrogate)
savefig(p, base_dir * "../trained_surrogates/sequence/latent_comp_linear_17_swap_prob=$swap_prob.png")
savefig(p, base_dir * "../trained_surrogates/sequence/latent_comp_linear_17_swap_prob=$swap_prob.pdf")
savefig(p, base_dir * "../trained_surrogates/sequence/latent_comp_linear_17_swap_prob=$swap_prob.svg")


