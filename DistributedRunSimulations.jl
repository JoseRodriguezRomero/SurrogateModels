using Distributed

using Pkg
Pkg.activate(".")
using QuantumSavory

num_workers = 16
if length(ARGS) >= 1
    num_workers = parse(Int, ARGS[1])
end

if nprocs() < num_workers
    addprocs(num_workers - nprocs(), exeflags=["--project", "--compiled-modules=yes"])
end

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using QuantumSavory
    Pkg.develop(path="../QuantumSavory.jl/")
    Pkg.update()
    # using QuantumSavory.ProtocolZoo
end

@everywhere begin
    Pkg.develop(path="../QuantumSavory.jl/")
    include("setup.jl")
    Pkg.update()
end

@everywhere include("setup.jl")
@everywhere sample_fidelities = 0.50:0.05:0.95

@everywhere my_id = myid()
@everywhere num_workers = nprocs()

@everywhere for F in sample_fidelities
    base_dir = "saved_data/F_$(F)/"
    if !isdir(base_dir)
        if my_id == 1
            mkdir(base_dir)
        end

        sleep(0.1)
    end

    error_dir = base_dir * "error_logs/"
    if !isdir(error_dir)
        if my_id == 1
            mkdir(error_dir)
        end

        sleep(0.1)
    end
end

@everywhere for F in sample_fidelities
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

    base_dir = "saved_data/F_$(F)/"
    error_dir = base_dir * "error_logs/"
    error_log_file = open(error_dir * "error_log_thread_$(my_id).txt", "w")

    obs1_vec = zeros(Float64, 0)
    obs2_vec = zeros(Float64, 0)
    t_vec = zeros(Float64, 0)

    for sample_id in my_id:num_workers:n_sample_files
        sim, network = simulation_setup(sizes, T2)
        registers = [network[node] for node in vertices(network)]

        for (; src, dst) in edges(network)
            @process EntanglerProt(sim, network, src, dst,
                pairstate=noisy_pair_func(F),
                retry_lock_time=entangler_wait_time,
                local_busy_time_post=entangler_busy_time, margin=5)()

            # @process EntanglerProt(sim, network, src, dst, margin = 5)()
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
                    nodeH=(>(node)), nodeL=(<(node)))()
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

        # while true
        #     try
        #         run(sim, 1000.0)
        #         sleep(1.0)
        #         break
        #     catch e
        #         @warn "Simulation (F = $(F)) sample $(sample_id) on thread $(thread_id) terminated with an error. Retrying..."

        #         println(error_log_files[thread_id], "Simulation (F = $(F)) sample $(sample_id) on thread $(thread_id) terminated with an error. Retrying...")
        #         println(error_log_files[thread_id], e)

        #         if thread_id == 1
        #             @warn e
        #             @warn stacktrace()
        #         end

        #         sleep(1.0)
        #     end
        # end

        run(sim, 1000.0)

        savefile = base_dir * "sample_$(sample_id)_thread_$(thread_id).h5"
        metadata = Dict(
            "sizes" => sizes,
            "T2" => T2,
            "F" => F,
            "entangler_wait_time" => entangler_wait_time,
            "entangler_busy_time" => entangler_busy_time,
            "swapper_wait_time" => swapper_wait_time,
            "swapper_busy_time" => swapper_busy_time,
            "purifier_wait_time" => purifier_wait_time,
            "purifier_busy_time" => purifier_busy_time,
            "n_sample_files" => n_sample_files
        )
        save(savefile, consumer_prot; metadata)
    end

    close(error_log_file)
end

