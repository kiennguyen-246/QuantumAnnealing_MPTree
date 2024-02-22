import json
import os
import re
import csv

import numpy as np

ROOT = os.getcwd()
file_list = os.listdir(ROOT)
input_dir = ROOT + '/output/'
output_dir = ROOT + '/output/stats/'


# Finding best chain strength
def exp1():
    print("Running statistics for Quantum Annealing and Simulated Annealing comparison...")
    stats = {
        "num_vars": {
            "L": [],
            "F": [],
            "N": [],
        },
        "num_qubit": {
            "L": [],
            "F": [],
            "N": [],
        },
        "non_zero": {
            "L": [],
            "F": [],
            "N": [],
        },
        "avg_chain_length": {
            "L": [],
            "F": [],
            "N": [],
        },
        "max_chain_length": {
            "L": [],
            "F": [],
            "N": [],
        },
        "time_elapsed": {
            "L": [],
            "F": [],
            "N": [],
        },
        "success_rate": {
            "L": [],
            "F": [],
            "N": [],
        },
        "optimal_rate": {
            "L": [],
            "F": [],
            "N": [],
        },
    }
    problem_sizes = []
    for directory in os.listdir(input_dir):
        if ".phy" not in directory:
            continue
        # if directory[5] not in {'1', '2'}:
        #     continue
        print(directory)
        input_dir2 = input_dir + directory + '/'
        f_dicts = [{}, {}]
        l_dicts = [{}, {}]
        n_dicts = [{}, {}]
        for file in os.listdir(input_dir2):
            if "json" not in file:
                continue
            if "SA" in file or "map" in file:
                continue
            file_type = 0
            if file[-7] == "_":
                file_type = int(file[-6]) - 1
            if file[0] == "L":
                with open(input_dir2 + file, "r") as f:
                    if file[-7] != "_":
                        embedding_config = json.load(f)
                        avg_chain_length = np.mean([len(embedding_config[vertex]) for vertex in embedding_config])
                        l_dicts[0]["avg_chain_length"] = avg_chain_length
                    else:
                        load = json.load(f)
                        for key in load:
                            l_dicts[file_type][key] = load[key]
            elif file[0] == "F":
                with open(input_dir2 + file, "r") as f:
                    if file[-7] != "_":
                        embedding_config = json.load(f)
                        avg_chain_length = np.mean([len(embedding_config[vertex]) for vertex in embedding_config])
                        f_dicts[0]["avg_chain_length"] = avg_chain_length
                    else:
                        load = json.load(f)
                        for key in load:
                            f_dicts[file_type][key] = load[key]
            else:
                with open(input_dir2 + file, "r") as f:
                    if file[-7] != "_":
                        embedding_config = json.load(f)
                        avg_chain_length = np.mean([len(embedding_config[vertex]) for vertex in embedding_config])
                        n_dicts[0]["avg_chain_length"] = avg_chain_length
                    else:
                        load = json.load(f)
                        for key in load:
                            n_dicts[file_type][key] = load[key]
                if n_dicts[1] is not None:
                    if "chain_strength_prefactor" in n_dicts[0] and n_dicts[0]["chain_strength_prefactor"] != 0.3:
                        continue
            print(file)
        problem_size = int(re.split("_", directory)[3].split(".")[0])
        problem_sizes.append(problem_size)
        stats["num_vars"]["L"].append(l_dicts[0]["num_vars"])
        stats["num_vars"]["F"].append(f_dicts[0]["num_vars"])
        stats["num_vars"]["N"].append(n_dicts[0]["num_vars"])
        stats["num_qubit"]["L"].append(l_dicts[0]["num_qubit"])
        stats["num_qubit"]["F"].append(f_dicts[0]["num_qubit"])
        stats["num_qubit"]["N"].append(n_dicts[0]["num_qubit"])
        stats["non_zero"]["L"].append(l_dicts[1]["non_zero"])
        stats["non_zero"]["F"].append(f_dicts[1]["non_zero"])
        stats["non_zero"]["N"].append(n_dicts[1]["non_zero"])
        stats["avg_chain_length"]["L"].append(l_dicts[0]["avg_chain_length"])
        stats["avg_chain_length"]["F"].append(f_dicts[0]["avg_chain_length"])
        stats["avg_chain_length"]["N"].append(n_dicts[0]["avg_chain_length"])
        stats["max_chain_length"]["L"].append(l_dicts[0]["max_chain_length"])
        stats["max_chain_length"]["F"].append(f_dicts[0]["max_chain_length"])
        stats["max_chain_length"]["N"].append(n_dicts[0]["max_chain_length"])
        stats["time_elapsed"]["L"].append(l_dicts[0]["time_elapsed"])
        stats["time_elapsed"]["F"].append(f_dicts[0]["time_elapsed"])
        stats["time_elapsed"]["N"].append(n_dicts[0]["time_elapsed"])
        stats["success_rate"]["L"].append(l_dicts[1]["success_rate"] / 1000)
        stats["success_rate"]["F"].append(f_dicts[1]["success_rate"] / 1000)
        stats["success_rate"]["N"].append(n_dicts[1]["success_rate"] / 1000)
        stats["optimal_rate"]["L"].append(l_dicts[1]["optimal_rate"] / 1000)
        stats["optimal_rate"]["F"].append(f_dicts[1]["optimal_rate"] / 1000)
        stats["optimal_rate"]["N"].append(n_dicts[1]["optimal_rate"] / 1000)
    ans1 = {
        "num_vars": {
            "L": {
                "mean": np.mean(stats["num_vars"]["L"]),
                "std": np.std(stats["num_vars"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["num_vars"]["F"]),
                "std": np.std(stats["num_vars"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["num_vars"]["N"]),
                "std": np.std(stats["num_vars"]["N"]),
            },
        },
        "num_qubit": {
            "L": {
                "mean": np.mean(stats["num_qubit"]["L"]),
                "std": np.std(stats["num_qubit"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["num_qubit"]["F"]),
                "std": np.std(stats["num_qubit"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["num_qubit"]["N"]),
                "std": np.std(stats["num_qubit"]["N"]),
            },
        },
        "non_zero": {
            "L": {
                "mean": np.mean(stats["non_zero"]["L"]),
                "std": np.std(stats["non_zero"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["non_zero"]["F"]),
                "std": np.std(stats["non_zero"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["non_zero"]["N"]),
                "std": np.std(stats["non_zero"]["N"]),
            },
        },
        "avg_chain_length": {
            "L": {
                "mean": np.mean(stats["avg_chain_length"]["L"]),
                "std": np.std(stats["avg_chain_length"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["avg_chain_length"]["F"]),
                "std": np.std(stats["avg_chain_length"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["avg_chain_length"]["N"]),
                "std": np.std(stats["avg_chain_length"]["N"]),
            },
        },
        "max_chain_length": {
            "L": {
                "mean": np.mean(stats["max_chain_length"]["L"]),
                "std": np.std(stats["max_chain_length"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["max_chain_length"]["F"]),
                "std": np.std(stats["max_chain_length"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["max_chain_length"]["N"]),
                "std": np.std(stats["max_chain_length"]["N"]),
            },
        },
        "time_elapsed": {
            "L": {
                "mean": np.mean(stats["time_elapsed"]["L"]),
                "std": np.std(stats["time_elapsed"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["time_elapsed"]["F"]),
                "std": np.std(stats["time_elapsed"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["time_elapsed"]["N"]),
                "std": np.std(stats["time_elapsed"]["N"]),
            },
        },
        "success_rate": {
            "L": {
                "mean": np.mean(stats["success_rate"]["L"]),
                "std": np.std(stats["success_rate"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["success_rate"]["F"]),
                "std": np.std(stats["success_rate"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["success_rate"]["N"]),
                "std": np.std(stats["success_rate"]["N"]),
            },
        },
        "optimal_rate": {
            "L": {
                "mean": np.mean(stats["optimal_rate"]["L"]),
                "std": np.std(stats["optimal_rate"]["L"]),
            },
            "F": {
                "mean": np.mean(stats["optimal_rate"]["F"]),
                "std": np.std(stats["optimal_rate"]["F"]),
            },
            "N": {
                "mean": np.mean(stats["optimal_rate"]["N"]),
                "std": np.std(stats["optimal_rate"]["N"]),
            },
        },
    }
    with open(output_dir + "exp1.json", "w") as f:
        json.dump(ans1, f, indent=4)
    with open(output_dir + "num_qubits.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem Size", "#Qubits", "Formulation"])
        for i in range(len(problem_sizes)):
            writer.writerow([problem_sizes[i], stats["num_qubit"]["L"][i], "L"])
            writer.writerow([problem_sizes[i], stats["num_qubit"]["F"][i], "F"])
            writer.writerow([problem_sizes[i], stats["num_qubit"]["N"][i], "N"])
    with open(output_dir + "avg_chain_length.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem Size", "ECLen", "Formulation"])
        for i in range(len(problem_sizes)):
            writer.writerow([problem_sizes[i], stats["avg_chain_length"]["L"][i], "L"])
            writer.writerow([problem_sizes[i], stats["avg_chain_length"]["F"][i], "F"])
            writer.writerow([problem_sizes[i], stats["avg_chain_length"]["N"][i], "N"])
    with open(output_dir + "success_rate.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem Size", "%Sol", "Formulation"])
        for i in range(len(problem_sizes)):
            writer.writerow([problem_sizes[i], stats["success_rate"]["L"][i] * 100, "L"])
            writer.writerow([problem_sizes[i], stats["success_rate"]["F"][i] * 100, "F"])
            writer.writerow([problem_sizes[i], stats["success_rate"]["N"][i] * 100, "N"])


    #
    # df_array = [["L", "F", "N"]]
    # for key in stats["num_qubit"]:



# Finding best chain strength
def exp2():
    print("Running statistics for best chain strength...")
    stats = {
        "time_elapsed": {},
        "success_rate": {},
        "optimal_rate": {},
    }
    for prefactor in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        stats["time_elapsed"][prefactor] = []
        stats["success_rate"][prefactor] = []
        stats["optimal_rate"][prefactor] = []
    for directory in os.listdir(input_dir):
        if ".phy" not in directory:
            continue
        if directory[5] not in {'1', '2'}:
            continue
        print(directory)
        input_dir2 = input_dir + directory + '/'
        test_stat_dict = {}
        for prefactor in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            test_stat_dict[prefactor] = [{}, {}]
        prefactor = 0
        for file in os.listdir(input_dir2):
            if file[0] != 'N':
                continue
            if file[-7] != "_" or "json" not in file:
                continue
            file_type = int(file[-6]) - 1
            if "SA" in file:
                continue
            with open(input_dir2 + file, "r") as f:
                tmp = json.load(f)
                if "chain_strength_prefactor" in tmp:
                    prefactor = tmp["chain_strength_prefactor"]
                test_stat_dict[prefactor][file_type] = tmp
            print(file)
        for prefactor in test_stat_dict:
            time_elapsed = test_stat_dict[prefactor][0]["time_elapsed"]
            success_rate = test_stat_dict[prefactor][1]["success_rate"] / 1000
            optimal_rate = test_stat_dict[prefactor][1]["optimal_rate"] / 1000
            stats["time_elapsed"][prefactor].append(time_elapsed)
            stats["success_rate"][prefactor].append(success_rate)
            stats["optimal_rate"][prefactor].append(optimal_rate)

    ans2 = {
        "time_elapsed": {},
        "success_rate": {},
        "optimal_rate": {},
    }
    for prefactor in stats["time_elapsed"]:
        ans2["time_elapsed"][prefactor] = {
            "mean": np.mean(stats["time_elapsed"][prefactor]),
            "std": np.std(stats["time_elapsed"][prefactor]),
        }
        ans2["success_rate"][prefactor] = {
            "mean": np.mean(stats["success_rate"][prefactor]),
            "std": np.std(stats["success_rate"][prefactor]),
        }
        ans2["optimal_rate"][prefactor] = {
            "mean": np.mean(stats["optimal_rate"][prefactor]),
            "std": np.std(stats["optimal_rate"][prefactor]),
        }

    with open(output_dir + "exp2.json", "w") as f:
        json.dump(ans2, f, indent=4)


# Comparing QA vs SA
def exp3():
    print("Running statistics for Quantum Annealing and Simulated Annealing comparison...")
    stats = {
        "time_elapsed": {
            "SA": [],
            "QA": [],
        },
        "success_rate": {
            "SA": [],
            "QA": [],
        },
        "optimal_rate": {
            "SA": [],
            "QA": [],
        },
    }
    for directory in os.listdir(input_dir):
        if ".phy" not in directory:
            continue
        print(directory)
        input_dir2 = input_dir + directory + '/'
        sa_dicts = [{}, {}]
        qa_dicts = [{}, {}]
        for file in os.listdir(input_dir2):
            if file[0] != 'N':
                continue
            if file[-7] != "_" or "json" not in file:
                continue
            file_type = int(file[-6]) - 1
            if "SA" in file:
                with open(input_dir2 + file, "r") as f:
                    sa_dicts[file_type] = json.load(f)
            else:
                with open(input_dir2 + file, "r") as f:
                    qa_dicts[file_type] = json.load(f)
                if qa_dicts[1] is not None:
                    if "chain_strength_prefactor" in qa_dicts[0] and qa_dicts[0]["chain_strength_prefactor"] != 0.3:
                        continue
            print(file)
        stats["time_elapsed"]["SA"].append(sa_dicts[0]["time_elapsed"])
        stats["time_elapsed"]["QA"].append(qa_dicts[0]["time_elapsed"])
        stats["success_rate"]["SA"].append(sa_dicts[1]["success_rate"] / 1000)
        stats["success_rate"]["QA"].append(qa_dicts[1]["success_rate"] / 1000)
        stats["optimal_rate"]["SA"].append(sa_dicts[1]["optimal_rate"] / 1000)
        stats["optimal_rate"]["QA"].append(qa_dicts[1]["optimal_rate"] / 1000)
    ans3 = {
        "time_elapsed": {
            "SA": {
                "mean": np.mean(stats["time_elapsed"]["SA"]),
                "std": np.std(stats["time_elapsed"]["SA"]),
            },
            "QA": {
                "mean": np.mean(stats["time_elapsed"]["QA"]),
                "std": np.std(stats["time_elapsed"]["QA"]),
            },
        },
        "success_rate": {
            "SA": {
                "mean": np.mean(stats["success_rate"]["SA"]),
                "std": np.std(stats["success_rate"]["SA"]),
            },
            "QA": {
                "mean": np.mean(stats["success_rate"]["QA"]),
                "std": np.std(stats["success_rate"]["QA"]),
            },
        },
        "optimal_rate": {
            "SA": {
                "mean": np.mean(stats["optimal_rate"]["SA"]),
                "std": np.std(stats["optimal_rate"]["SA"]),
            },
            "QA": {
                "mean": np.mean(stats["optimal_rate"]["QA"]),
                "std": np.std(stats["optimal_rate"]["QA"]),
            },
        },
    }
    with open(output_dir + "exp3.json", "w") as f:
        json.dump(ans3, f, indent=4)


exp1()
# exp3()
