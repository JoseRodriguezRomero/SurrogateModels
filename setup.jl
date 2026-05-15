# For convenient graph data structures
using Graphs

# For discrete event simulation
using ResumableFunctions
using ResumableFunctions: @resumable
using ConcurrentSim
using ConcurrentSim: @process

# The workhorse for the simulation
using QuantumSavory
using QuantumSavory.ProtocolZoo

# For saving the log of the simulation
using HDF5, FileIO, CSV, DataFrames

# Predefined useful circuits
using QuantumSavory.CircuitZoo: EntanglementSwap, Purify2to1

using Distributions

"""Creates the datastructures representing the simulated network"""
function simulation_setup(
    sizes, # Array giving the number of qubits in each node
    T2 # T2 dephasing times for the qubits
    ;
    representation = QuantumOpticsRepr # Representation to use for the qubits
    )
    R = length(sizes) # Number of registers

    # All of the quantum register we will be simulating
    registers = Register[]
    for s in sizes
        traits = [Qubit() for _ in 1:s]
        repr = [representation() for _ in 1:s]
        # bg = [Depolarization(T2) for _ in 1:s]
        bg = [T2Dephasing(T2) for _ in 1:s]
        # bg = [PauliNoise(0.2,0.2,0.2) for _ in 1:s]
        push!(registers, Register(traits,repr,bg))
    end

    # A graph structure defining the connectivity among registers
    # It is not necessary to use such a structure, however, it is a convenient way to
    # store data about the simulation (and we have created helper plotting functions
    # expecting such a structure).
    graph = grid([R])
    network = RegisterNet(graph, registers) # A graphs with extra "meta data"

    # The scheduler datastructure for the discrete event simulation
    sim = get_time_tracker(network)

    # Add a register datastructures and event locks to each node.
    for v in vertices(network)
        # Create an array specifying whether a qubit is entangled with another qubit
        network[v,:enttrackers] = Any[nothing for i in 1:sizes[v]]
    end

    sim, network
end

function noisy_pair_func(F)
    perfect_pair = StabilizerState("ZZ XX")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(perfect_pair_dm)

    return F*perfect_pair_dm + (1-F)*mixed_dm # TODO make a depolarization helper
end

const XX = X⊗X
const ZZ = Z⊗Z
const YY = Y⊗Y

@resumable function purifier(
    sim::Environment,  # The scheduler for all simulation events
    network,           # The graph of quantum nodes
    nodea,             # One of the nodes on which the pairs to be purified rest
    nodeb,             # The other such node
    purifier_wait_time,# The wait time in case there are no pairs available for purification
    purifier_busy_time # The duration of the purification circuit
    )
    round = 0
    while true
        pairs_of_bellpairs = findqubitstopurify(network,nodea,nodeb)
        if isnothing(pairs_of_bellpairs)
            @yield timeout(sim, purifier_wait_time)
            continue
        end
        pair1qa, pair1qb, pair2qa, pair2qb = pairs_of_bellpairs
        locks = [network[nodea][[pair1qa,pair2qa]];
                 network[nodeb][[pair1qb,pair2qb]]]
        @yield mapreduce(request, &, locks)
        @yield timeout(sim, purifier_busy_time)
        rega = network[nodea]
        regb = network[nodeb]
        purifyerror =  (:X, :Z)[round%2+1]
        purificationcircuit = Purify2to1(purifyerror)
        success = purificationcircuit(rega[pair1qa],regb[pair1qb],rega[pair2qa],regb[pair2qb])
        if !success
            network[nodea,:enttrackers][pair1qa] = nothing
            network[nodeb,:enttrackers][pair1qb] = nothing
            @simlog sim "failed purification at $(nodea):$(pair1qa)&$(pair2qa) and $(nodeb):$(pair1qb)&$(pair2qb)"
        else
            round += 1
            @simlog sim "purification at $(nodea):$(pair1qa) $(nodeb):$(pair1qb) by sacrifice of $(nodea):$(pair1qa) $(nodeb):$(pair1qb)"
        end
        network[nodea,:enttrackers][pair2qa] = nothing
        network[nodeb,:enttrackers][pair2qb] = nothing
        release.(locks)
    end
end

function findqubitstopurify(network,nodea,nodeb)
    enttrackers = network[nodea,:enttrackers]
    rega = network[nodea]
    regb = network[nodeb]
    enttrackers = [(i=i,n...) for (i,n) in enumerate(enttrackers)
                   if !isnothing(n) && n.node==nodeb && !islocked(rega[i]) && !islocked(regb[n.slot])]
    if length(enttrackers)>=2
        aqubits = [n.i for n in enttrackers[end-1:end]]
        bqubits = [n.slot for n in enttrackers[end-1:end]]
        return aqubits[2], bqubits[2], aqubits[1], bqubits[1]
    else
        return nothing
    end
end

