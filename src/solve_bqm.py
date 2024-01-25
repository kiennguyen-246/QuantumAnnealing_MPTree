import os
import time

from dwave.system import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.embedding.chain_strength import uniform_torque_compensation
from neal import SimulatedAnnealingSampler
import dwave.inspector

from src.export_embedding import get_embedding

data_name = "sequences"
output_dir = "output/" + data_name + "/"


def solve_quantum_annealing(bqm,
                            method="?_",
                            num_reads=1000):
    chain_strength_prefactor = 0.3
    annealing_time = 200
    anneal_schedule_id = -1
    chain_strength = uniform_torque_compensation(
        bqm=bqm, prefactor=chain_strength_prefactor)
    sampler = EmbeddingComposite(DWaveSampler())
    schedules = {12: [(0.0, 0.0), (40.0, 0.4), (180.0, 0.4), (200.0, 1.0)],
                 11: [(0.0, 0.0), (40.0, 0.5), (120.0, 0.5), (200.0, 1.0)],
                 13: [(0.0, 0.0), (40.0, 0.5), (130.0, 0.5), (200.0, 1.0)],
                 14: [(0.0, 0.0), (30.0, 0.5), (160.0, 0.5), (200.0, 1.0)]}
    config = method + str(num_reads) + "-" + str(chain_strength) + "s" + str(anneal_schedule_id) + "_A" + str(
        annealing_time)
    # get_embedding(bqm, output_dir + config + ".json")
    start = time.time()
    if anneal_schedule_id == -1:
        response = sampler.sample(bqm=bqm,
                                  chain_strength=chain_strength,
                                  num_reads=num_reads,
                                  label=config,
                                  annealing_time=annealing_time)
    else:
        schedule = schedules[anneal_schedule_id]
        response = sampler.sample(bqm=bqm,
                                  chain_strength=chain_strength,
                                  num_reads=num_reads,
                                  label=config,
                                  anneal_schedule=schedule)
    end = time.time()
    dwave.inspector.show(response)
    chains = response.info["embedding_context"]["embedding"].values()
    for chain in chains:
        if len(chain) > 10:
            print(chain)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_response_output = open(output_dir + config + ".txt", "w")
    file_response_output.write("Config: " + config + "\n")
    file_response_output.write("Number of source variables: " + str(len(bqm.variables)) + "\n")
    file_response_output.write("Number of target variables: " + str(len(response.variables)) + "\n")
    file_response_output.write("Time Elapsed: " + str(end - start) + "\n")
    file_response_output.write(
        "Best State: " + str(response.record.sample[0]) + "\n" + str(response.record.energy[0]) + "\t" + str(
            response.record.chain_break_fraction[0]) + "\n")
    file_response_output.write("ChainStr/ChainLen: " + str(chain_strength) + "/" + str(
        max([len(chain) for chain in chains])) + "\n")
    file_response_output.write("Info: " + str(response.info["timing"]) + "\n")
    file_response_output.write("Embedding Info: " + str(response.info["embedding_context"]) + "\n")

    return response


def solve_simulated_annealing(bqm, method="?_", num_reads=1000):
    sampler = SimulatedAnnealingSampler()
    beta_range = [0.1, 4]
    num_sweeps = 1000
    config = method + str(num_reads) + "-SA" + "".join(str(beta_range).split(" ")) + "s" + str(num_sweeps)
    print(config)
    response = sampler.sample(bqm,
                              num_reads=num_reads,
                              label=config,
                              beta_range=beta_range,
                              num_sweeps=num_sweeps)
    return response
