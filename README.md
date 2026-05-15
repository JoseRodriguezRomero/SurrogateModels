# SurrogateModels using QuatumSavory

These scripts are inteded for generating simple repeater chain data to then 
train/develop a more convenient surrogate model. These scripts run several 
instances of a simple repeater chain where the number of nodes is `n_nodes` 
and the number of qubits per node is `qubits_per_node`.

Thes simulations are ran by setting different fidelities when initialzing the
qubits of the nodes. For each of the fidelities used when initializing the 
qubits, 350 simulations are ran with 1000.0 seconds of simulation time to 
gather enough statistics. The recoded data is then saved in HDF5 format for 
convenience.

To accelerate the gathering of data, multithreading can be used, and to ran 
the simulations using `N` threads run:
```bash
julia --threads N RunSimulations.jl
```