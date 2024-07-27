import gc
import os
import shutil
import time

import numpy as np
import pandas as pd
import pynvml
import torch

from algorithms.DeepIF import DeepIF
from data.DatasetsSettings import DatasetsSettings
from utils.MemoryMonitor import MemoryMonitor

pynvml.nvmlInit()

def reset_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
def force_gc():
    gc.collect()

EXPERIMENTS_NUMBER = 100

if os.path.exists("results"):
    shutil.rmtree("results")
    os.makedirs("results")

for datasetName in DatasetsSettings.DATASETS_NAMES:
    readedData = pd.read_csv('datasets/' + datasetName["source"] + '.csv', sep=',', dtype=np.float64)
    dataXY = readedData.values
    xIn = dataXY[:, :len(dataXY[0]) - 1]
    yIn = dataXY[:, len(dataXY[0]) - 1:][:, 0]

    for optimization_flag in [True, False]:
        for experiment_id in range(EXPERIMENTS_NUMBER):
            np.random.seed()

            classifier = DeepIF(optimization=optimization_flag)
            algorithm_name = classifier.algorithm_name()

            print(algorithm_name + " - " + datasetName["name"] + ":")
            print("\tdimensionality = " + str(len(xIn[0])))
            print("\tsamples number = " + str(len(xIn)))

            reset_gpu_memory()
            force_gc()

            pid = os.getpid()
            monitor = MemoryMonitor(pid)
            monitor.start()

            classifier.fit(xIn)

            monitor.stop()
            monitor.join()

            ram_usage = monitor.max_ram_usage
            vram_usage = monitor.max_vram_usage
            print(f"\tmax RAM usage: {float(ram_usage) / (1024 ** 2):.2f} MB")
            print(f"\tmax VRAM usage: {float(vram_usage) / (1024 ** 2):.2f} MB")

            dir_train_ram = "results/memory/" + datasetName["source"] + "/train_ram"
            isExist = os.path.exists(dir_train_ram)
            if not isExist:
                os.makedirs(dir_train_ram)

            file_train_ram = open(dir_train_ram + "/" + algorithm_name + ".txt", "a")
            file_train_ram.write(str(ram_usage) + "\n")
            file_train_ram.close()

            dir_train_vram = "results/memory/" + datasetName["source"] + "/train_vram"
            isExist = os.path.exists(dir_train_vram)
            if not isExist:
                os.makedirs(dir_train_vram)

            file_train_vram = open(dir_train_vram + "/" + algorithm_name + ".txt", "a")
            file_train_vram.write(str(vram_usage) + "\n")
            file_train_vram.close()

            dir_train_cpu_stages = "results/time/" + datasetName["source"] + "/train_cpu"
            isExist = os.path.exists(dir_train_cpu_stages)
            if not isExist:
                os.makedirs(dir_train_cpu_stages)

            file_train_cpu_stages = open(dir_train_cpu_stages + "/" + algorithm_name + ".txt", "a")
            file_train_cpu_stages.write(str(classifier.cpu_stages_fit_time) + "\n")
            file_train_cpu_stages.close()

            print("\ttrain - cpu stage = " + str(classifier.cpu_stages_fit_time / 1000000.0) + " [ms]")

            dir_train_gpu_stages = "results/time/" + datasetName["source"] + "/train_gpu"
            isExist = os.path.exists(dir_train_gpu_stages)
            if not isExist:
                os.makedirs(dir_train_gpu_stages)

            file_train_gpu_stages = open(dir_train_gpu_stages + "/" + algorithm_name + ".txt", "a")
            file_train_gpu_stages.write(str(classifier.gpu_stages_fit_time) + "\n")
            file_train_gpu_stages.close()

            print("\ttrain - gpu stage = " + str(classifier.gpu_stages_fit_time / 1000000.0) + " [ms]")

            reset_gpu_memory()
            force_gc()

pynvml.nvmlShutdown()