using QuantumSavory
using QuantumSavory.ProtocolZoo
using ResumableFunctions
using ConcurrentSim

comm_slot, storage_slot = 1, 2

A = Register(2, T1Decay(1.0))
B = Register(2, T1Decay(1.0))
C = Register(2, T1Decay(1.0))
D = Register(2, T1Decay(1.0))
net = RegisterNet([A, B, C, D])

for register in net.registers
    initialize!(register[2], X1)
end

entangler_AB = EntanglerProt(get_time_tracker(net), net, 1, 2, 
    chooseA = comm_slot, chooseB = comm_slot, 
    pairstate = StabilizerState("ZX XZ"))

@process entangler_AB()

@resumable function fusion_protocol(sim::Simulation, node::Register)
    while true
        # Wait for a change in the classical
        # metadata stored in the register:
        @yield onchange_tag(node)

        # Someone just tagged a register
        # i.e. a Bell pair is ready.
        while true 
            # Loop until there are no pairs left.
            # Query whether there are entangled pairs:
            # (W stands for "Wildcard")
            entry = querydelete!(node, EntanglementCounterpart, W, W)
            
            if isnothing(entry)
                break
            end

            _, remote_node, _ = entry.tag

            # Prepare a quantum circuit and run it:
            circuit = Fusion()
            circuit(node, net[remote_node], comm_slot, storage_slot)
        end
    end
end

@process fusion_protocol(entangler_AB.sim, A)
@process fusion_protocol(entangler_AB.sim, C)
