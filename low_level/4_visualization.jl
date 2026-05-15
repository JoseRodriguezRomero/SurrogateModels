include("setup.jl")

using Base.Threads

for F in 0.50:0.05:0.95
    sizes = zeros(Int,5) .+ 10 # Number of qubits in each register
    T2 = 100.0                 # T2 dephasing time of all qubits
    # F = 0.85                 # Fidelity of the raw Bell pairs
    entangler_wait_time = 0.1  # How long to wait if all qubits are busy before retrying entangling
    entangler_busy_time = 1.0  # How long it takes to establish a newly entangled pair
    swapper_wait_time = 0.1    # How long to wait if all qubits are unavailable for swapping
    swapper_busy_time = 0.15   # How long it takes to swap two qubits
    purifier_wait_time = 0.15  # How long to wait if there are no pairs to be purified
    purifier_busy_time = 0.2   # How long the purification circuit takes to execute 
    n_sample_files = 150       # How many files to save for each F value (should be <= n_threads)
    
    n_threads = Threads.nthreads()
    obs1_vecs = Dict{Int, Any}()
    obs2_vecs = Dict{Int, Any}()
    t_vecs = Dict{Int, Any}()
    for thread_id in 1:n_threads
        obs1_vecs[thread_id] = zeros(Float64,0)
        obs2_vecs[thread_id] = zeros(Float64,0)
        t_vecs[thread_id] = zeros(Float64,0)
    end

    dir = "low_level/saved_data/F_$(F)/"
    if !isdir(dir)
        mkdir(dir)
    end

    function save_data(save_dir::String, thread_id::Int, sample_id::Int)
        function save_configuration(file_name::String)
            open(file_name, "w") do io
                println(io, "sizes = ", sizes)
                println(io, "T2 = ", T2)
                println(io, "F = ", F)
                println(io, "entangler_wait_time = ", entangler_wait_time)
                println(io, "entangler_busy_time = ", entangler_busy_time)
                println(io, "swapper_wait_time = ", swapper_wait_time)
                println(io, "swapper_busy_time = ", swapper_busy_time)
                println(io, "purifier_wait_time = ", purifier_wait_time)
                println(io, "purifier_busy_time = ", purifier_busy_time)
                println(io, "n_sample_files = ", n_sample_files)
            end
        end

        function save_observations(file_name::String, 
            obs1_vec::Vector{Float64}, obs2_vec::Vector{Float64}, 
            t_vec::Vector{Float64})
            open(file_name, "w") do io
                for i in eachindex(obs1_vec)
                    println(io, obs1_vec[i], " ", obs2_vec[i], " ", t_vec[i])
                end
            end
        end

        if sample_id == 1
            save_configuration(save_dir * "config.txt")
        end

        save_observations(save_dir * "observations_sample_$(sample_id).txt", 
            obs1_vecs[thread_id], obs2_vecs[thread_id], t_vecs[thread_id])
    end

    @threads for thread_id in 1:n_threads
        for sample_id in thread_id:n_threads:n_sample_files
            sim, network = simulation_setup(sizes, T2)
            registers = [network[node] for node in vertices(network)]

            for (;src, dst) in edges(network)
                # @process entangler(sim, network, src, dst, noisy_pair_func(F), 
                #     entangler_wait_time, entangler_busy_time)
                @process EntanglerProt(sim, network, src, dst, 
                    pairstate = noisy_pair_func(F), retry_lock_time = entangler_wait_time, 
                    local_busy_time_post = entangler_busy_time, margin = 5)()
                # @process EntanglerProt(sim, network, src, dst, margin = 5)()
            end
            for node in vertices(network)
                # @process SwapperProt(sim, network, node, 
                # retry_lock_time = swapper_wait_time,
                # local_busy_time = swapper_busy_time)()
                # @process SwapperProt(sim, network, node)()

                if node > 1 || node < nv(network)
                    @process SwapperProt(sim, network, node, 
                        nodeH = >(node), nodeL = <(node))()
                end
                # @process swapper(sim, network, node, swapper_wait_time, swapper_busy_time)
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
            entanglement_consumer = EntanglementConsumer(sim, network, 1, length(registers))
            @process entanglement_consumer()
            run(sim, 10000.0)

            obs1_vec = zeros(Float64,length(entanglement_consumer._log))
            obs2_vec = zeros(Float64,length(entanglement_consumer._log))
            t_vec = zeros(Float64,length(entanglement_consumer._log))
            for i in eachindex(entanglement_consumer._log)
                obs1_vec[i] = entanglement_consumer._log[i].obs1
                obs2_vec[i] = entanglement_consumer._log[i].obs2
                t_vec[i] = entanglement_consumer._log[i].t
            end

            obs1_vecs[thread_id] = obs1_vec
            obs2_vecs[thread_id] = obs2_vec
            t_vecs[thread_id] = t_vec

            save_data("low_level/saved_data/F_$(F)/", thread_id, sample_id)
        end        
    end
end

