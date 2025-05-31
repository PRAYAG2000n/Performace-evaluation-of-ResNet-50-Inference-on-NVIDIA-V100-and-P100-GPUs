import subprocess
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import time
import os
print("TF Graph Execution")

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_async'
tf.get_logger().setLevel('ERROR')

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Setup for GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
gpu_stats_list = []
@tf.function
def infer(batch_x):
    return model(batch_x, training=False)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test[:1000]  # Optional: limit test set to 1000 samples
y_test = y_test[:1000]
x_test = tf.image.resize(x_test, [224, 224]) / 255.0
y_test = tf.squeeze(y_test)

# Load pre-trained ResNet-50
model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=True)

# Function to get GPU stats
def get_gpu_stats():
    try:
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.free,memory.total',
            '--format=csv,nounits,noheader'
        ])
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        print(f"Error calling nvidia-smi: {e}")
        return None

# Benchmarking function
def benchmark_batch_size(batch_size):
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    total_correct = 0
    total_images = x_test.shape[0]
    gpu_stats_list.clear()

    print(f"\n--- Benchmarking ResNet-50 | Batch size: {batch_size} | Samples: {total_images} ---")

    start_time = time.time()
    for i, (batch_x, batch_y) in enumerate(test_ds):
        preds = infer(batch_x)
        pred_classes = tf.argmax(preds, axis=1)
        batch_y = tf.cast(batch_y, tf.int64)
        total_correct += tf.reduce_sum(tf.cast(pred_classes == batch_y, tf.int32)).numpy()

        # Capture GPU stats every 10 iterations
        if i % 100 == 0:
            stats = get_gpu_stats()
            if stats:
                gpu_stats_list.append(stats)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = total_images / total_time
    accuracy = total_correct / total_images * 100

    # Try to get current GPU memory
    try:
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        memory_used = memory_info['current'] / 1e6
    except AttributeError:
        memory_used = "N/A"
        print("TensorFlow version doesn't support 'get_memory_info'.")

    # Calculate average GPU stats
    if gpu_stats_list:
        gpu_util = []
        mem_used = []
        mem_free = []
        mem_total = []

        for stat in gpu_stats_list:
            util, used, free, total = map(float, stat.split(','))
            gpu_util.append(util)
            mem_used.append(used)
            mem_free.append(free)
            mem_total.append(total)

        avg_util = np.mean(gpu_util)
        avg_mem_used = np.mean(mem_used)
        avg_mem_free = np.mean(mem_free)
        avg_mem_total = np.mean(mem_total)
    else:
        avg_util = avg_mem_used = avg_mem_free = avg_mem_total = "N/A"

    # Print results
    print(f"\n--- Results for Batch Size {batch_size} ---")
    print(f"Total inference time     : {total_time:.2f} sec")
    print(f"Throughput               : {throughput:.2f} images/sec")
    print(f"Top-1 Accuracy           : {accuracy:.2f} %")
    print(f"GPU Memory (current)     : {memory_used} MB")

    print("\nAverage GPU Stats During Execution:")
    print(f"GPU Utilization          : {avg_util} %")
    print(f"Memory Used              : {avg_mem_used} MB")
    print(f"Memory Free              : {avg_mem_free} MB")
    print(f"Total Memory             : {avg_mem_total} MB")

# Run benchmark for each batch size
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
for bs in batch_sizes:
    benchmark_batch_size(bs)
