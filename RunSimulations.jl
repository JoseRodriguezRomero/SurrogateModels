using Pkg
using Base.Threads

Pkg.activate(".")

include("setup.jl")

n_threads = nthreads()
sample_fidelities = 0.50:0.05:0.95

@info("Using $(n_threads) threads.")

for F in sample_fidelities
    sizes = zeros(Int,5) .+ 10 # Number of qubits in each register
    T2 = 100.0                 # T2 dephasing time of all qubits
    # F = 0.85                 # Fidelity of the raw Bell pairs
    entangler_wait_time = 0.1  # How long to wait if all qubits are busy before retrying entangling
    entangler_busy_time = 1.0  # How long it takes to establish a newly entangled pair
    swapper_wait_time = 0.1    # How long to wait if all qubits are unavailable for swapping
    swapper_busy_time = 0.15   # How long it takes to swap two qubits
    purifier_wait_time = 0.15  # How long to wait if there are no pairs to be purified
    purifier_busy_time = 0.2   # How long the purification circuit takes to execute 
    n_sample_files = 350       # How many files to save for each F value (should be <= n_threads)

    base_dir = "saved_data"
    if !isdir(base_dir)
        mkdir(base_dir)
    end

    base_dir = "saved_data/F_$(F)/"
    if !isdir(base_dir)
        mkdir(base_dir)
    end

    error_dir = base_dir * "error_logs/"
    if !isdir(error_dir)
        mkdir(error_dir)
    end

    error_log_files = Dict{Int,IO}()
    for thread_id in 1:n_threads
        error_log_files[thread_id] = open(error_dir * "error_log_thread_$(thread_id).txt", "w")
    end

    obs1_vecs = Dict{Int, Vector{Float64}}()
    obs2_vecs = Dict{Int, Vector{Float64}}()
    t_vecs = Dict{Int, Vector{Float64}}()
    for sample_id in 1:n_sample_files
        obs1_vecs[sample_id] = zeros(Float64,0)
        obs2_vecs[sample_id] = zeros(Float64,0)
        t_vecs[sample_id] = zeros(Float64,0)
    end

    @threads for thread_id in 1:n_threads
        for sample_id in thread_id:n_threads:n_sample_files
            sim, network = simulation_setup(sizes, T2)
            registers = [network[node] for node in vertices(network)]

            for (;src, dst) in edges(network)
                @process EntanglerProt(sim, network, src, dst, 
                    pairstate = noisy_pair_func(F), 
                    retry_lock_time = entangler_wait_time, 
                    local_busy_time_post = entangler_busy_time, margin = 5)()
            end
            for node in vertices(network)
                # if node > 1 || node < nv(network)
                #     @process SwapperProt(sim, network, node, 
                #         retry_lock_time = swapper_wait_time,
                #         local_busy_time = swapper_busy_time,
                #         nodeH = >(node), nodeL = <(node))()
                # end

                if node > 1 || node < nv(network)
                    @process SwapperProt(sim, network, node, 
                        nodeH = >(node), nodeL = <(node))()
                end
            end
            for node in vertices(network)
                @process EntanglementTracker(sim, network, node)()
            end
            # for nodea in vertices(network)
            #     for nodeb in vertices(network)
            #         if nodeb>nodea
            #             # @process purifier(sim, network, nodea, nodeb, purifier_wait_time, 
            #             #     purifier_busy_time)
            #         end
            #     end
            # end

            consumer_prot = EntanglementConsumer(sim, network, 1, length(registers))
            @process consumer_prot()

            run(sim, 1000.0)

            savefile = base_dir * "sample_$(sample_id).h5"
            metadata = Dict(
                # Optional metadata
                "sizes" => sizes,
                "T2" => T2,
                "F" => F,
                "entangler_wait_time" => entangler_wait_time,
                "entangler_busy_time" => entangler_busy_time,
                "swapper_wait_time" => swapper_wait_time,
                "swapper_busy_time" => swapper_busy_time,
                "purifier_wait_time" => purifier_wait_time,
                "purifier_busy_time" => purifier_busy_time,
                
                # Mandatory metadata
                "description" => "Simulation of a quantum network with $(nv(network)) nodes and $(ne(network)) edges, using the specified parameters.",
                "simulator" => "QuantumSavory.jl",
                "reference_state" => "bell_pair",
                "log_format" => "pauli_observables"
            )
            save(savefile, consumer_prot; metadata)
        end
    end

    for thread_id in 1:n_threads
        close(error_log_files[thread_id])
    end
end

