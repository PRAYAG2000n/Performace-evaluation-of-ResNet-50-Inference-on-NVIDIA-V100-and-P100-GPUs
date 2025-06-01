
# ResNet-50 GPU Inference Benchmarking on Northeastern Explorer Cluster

This is the step-by-step instructions for running ResNet-50 inference benchmarks using Python on the Northeastern University Explorer GPU cluster.

---

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Connecting to the Cluster](#2-connecting-to-the-cluster)
- [3. Preparing the Environment](#3-preparing-the-environment)
- [4. Submitting a GPU Job](#4-submitting-a-gpu-job)
- [5. Running the Benchmark](#5-running-the-benchmark)
- [6. Viewing Results](#6-viewing-results)
- [7. Example Benchmark Output](#7-example-benchmark-output)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

- MobaXterm (or any SSH client) installed locally
- An active Northeastern University Explorer cluster account
- Python script for ResNet-50 inference (e.g., `ResNet-50.py`)
- Basic familiarity with SLURM and Anaconda

---

## 2. Connecting to the Cluster

1. **Open MobaXterm** and start a new SSH session:
    - Host: `login.explorer.northeastern.edu`
    - Username: `<your_username>`
2. Log in using your Northeastern credentials.

---

## 3. Preparing the Environment

Once logged in:

```sh
module load cuda/
module load anaconda3
conda activate <your_conda_env>
```
- Replace `<your_conda_env>` with your environment name, e.g., `newnev`.
- If you do not have a conda environment, create one:
```sh
conda create -n newnev python=3.8
conda activate newnev
pip install torch torchvision tensorflow pillow
```

---

## 4. Submitting a GPU Job

Request an interactive GPU session using SLURM. For example, to request a **V100 GPU**:

```sh
srun -p courses-gpu --gres=gpu:v100-sxm2:1 --pty --time=03:00:00 /bin/bash
```
To request a **P100 GPU**:
```sh
srun -p courses-gpu --gres=gpu:p100:1 --pty --time=03:00:00 /bin/bash
```
It will then show **job has been allocated**

To verify this type this command:
```
nvidia-smi
```
It will be displayed in this form
![Screenshot 2025-04-06 200636](https://github.com/user-attachments/assets/c6eadb3f-ca0c-4113-9810-7fef921b589f)


## 5. Running the Benchmark

Run the Python script):

```sh
python ResNet-50.py
```
You should see output that reports inference time, throughput, accuracy, GPU memory, and utilization for different batch sizes.

---

## 6. Viewing Results

Sample output will look like this:

```
--- Benchmarking ResNet-50 | Batch size: 8 | Samples: 1000 ---
Total inference time     : 20.80 sec
Throughput               : 48.08 images/sec
Top-1 Accuracy           : 0.00 %
GPU Memory (current)     : 734.91 MB

Average GPU Stats During Execution:
GPU Utilization          : 4.0 %
Memory Used              : 2391.0 MB
Memory Free              : 30109.0 MB
Total Memory             : 32768.0 MB
```

The script can be run with different batch sizes to observe how throughput and utilisation change.

---

## 7. Example Benchmark Output

```
--- Benchmarking ResNet-50 | Batch size: 4 | Samples: 1000 ---
Total inference time     : 3.27 sec
Throughput               : 305.82 images/sec
GPU Utilization          : 49.0 %

--- Benchmarking ResNet-50 | Batch size: 32 | Samples: 1000 ---
Total inference time     : 2.10 sec
Throughput               : 475.90 images/sec
GPU Utilization          : 69.0 %
```

---

## 8. Troubleshooting

- **Import errors**: Ensure you have installed all required packages inside your conda environment.
- **File not found**: Check that your script and any input data are in your working directory.
- **Memory errors**: Reduce the batch size if you run into GPU memory issues.
- **Accuracy = 0%**: Ensure you are running inference on properly preprocessed and labeled validation/test images (random data will yield 0%).

---

## Notes

- You can modify the Python scripts to use PyTorch or TensorFlow as needed.
- Adjust the SLURM resource request based on GPU availability (`--gres=gpu:a100:1` for A100, etc.).
- For extended jobs, use SLURM batch scripts instead of interactive sessions.

---

## References

- [Explorer Cluster Documentation](http://rc.northeastern.edu/support)
- [PyTorch ResNet-50 Docs](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
- [TensorFlow ResNet Guide](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)

---
