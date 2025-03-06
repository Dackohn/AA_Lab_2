import time
import tracemalloc
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Heap Sort
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

# 2️⃣ Merge Sort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

# 3️⃣ Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 4️⃣ Bucket Sort
def bucket_sort(arr):
    if len(arr) == 0:
        return arr

    min_value = min(arr)
    max_value = max(arr)
    bucket_count = len(arr) // 10 or 1
    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = int((num - min_value) * (bucket_count - 1) // (max_value - min_value + 1))
        buckets[index].append(num)

    for i in range(len(buckets)):
        buckets[i] = quick_sort(buckets[i])

    return [num for bucket in buckets for num in bucket]


RUN_SIZE = 32

def insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def timsort(arr):
    n = len(arr)

    for i in range(0, n, RUN_SIZE):
        insertion_sort(arr, i, min(i + RUN_SIZE - 1, n - 1))

    size = RUN_SIZE
    while size < n:
        for left in range(0, n, 2 * size):
            mid = left + size - 1
            right = min(left + 2 * size - 1, n - 1)
            if mid < right:
                merge(arr, left, mid, right)
        size *= 2

    return arr

def merge(arr, left, mid, right):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1


def bucket_sort_timsort(arr):
    if len(arr) == 0:
        return arr

    min_value = min(arr)
    max_value = max(arr)
    bucket_count = len(arr) // 10 or 1
    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = int((num - min_value) * (bucket_count - 1) // (max_value - min_value + 1))
        buckets[index].append(num)

    for i in range(len(buckets)):
        buckets[i] = timsort(buckets[i])
        print(buckets[i])

    return [num for bucket in buckets for num in bucket]

def bucket_sort_insertion(arr):
    """Bucket Sort that uses only Insertion Sort."""
    if len(arr) == 0:
        return arr

    min_value = min(arr)
    max_value = max(arr)
    bucket_count = len(arr) // 10 or 1
    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = int((num - min_value) * (bucket_count - 1) // (max_value - min_value + 1))
        buckets[index].append(num)

    for i in range(len(buckets)):
        if buckets[i]:
            insertion_sort(buckets[i], 0, len(buckets[i]) - 1)

    return [num for bucket in buckets for num in bucket]


# 🔹 Function to Generate Different Input Types
def generate_data(size, case):
    if case == 'random':
        return [random.randint(1, 100000) for _ in range(size)]
    elif case == 'sorted':
        return list(range(size))
    elif case == 'reversed':
        return list(range(size, 0, -1))
    elif case == 'large_duplicates':
        return [random.choice(range(100)) for _ in range(size)]
    elif case == 'float':
        return [random.uniform(1, 100000) for _ in range(size)]

def measure_performance(sorting_function, input_sizes, data_type):
    time_results = []
    memory_results = []

    for size in input_sizes:
        execution_times = []
        memory_usages = []

        for _ in range(10):
            test_data = generate_data(size, data_type)
            test_data_copy = test_data.copy()

            start_time = time.time()
            sorting_function(test_data_copy)
            end_time = time.time()

            execution_times.append(end_time - start_time)

            tracemalloc.start()
            sorting_function(test_data)
            memory_used = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            memory_usages.append(memory_used)

        median_time = statistics.median(execution_times)
        median_memory = statistics.median(memory_usages)

        time_results.append(median_time)
        memory_results.append(median_memory)

        print(f"Size: {size} | Median Time: {median_time:.4f} sec | Median Memory: {median_memory / 1024:.2f} KB")

    return time_results, memory_results

# 📊 Function to Plot Performance
def plot_performance(input_sizes, time_results, memory_results, algorithm_name, data_type):
    plt.figure(figsize=(12, 5))

    # Plot Execution Time
    plt.subplot(1, 2, 1)
    plt.plot(input_sizes, time_results, marker="o", linestyle="-", color="b")
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"{algorithm_name} - Execution Time ({data_type})")
    plt.grid()

    # Plot Memory Usage
    plt.subplot(1, 2, 2)
    plt.plot(input_sizes, np.array(memory_results) / 1024, marker="s", linestyle="-", color="r")
    plt.xlabel("Input Size")
    plt.ylabel("Memory Usage (KB)")
    plt.title(f"{algorithm_name} - Memory Usage ({data_type})")
    plt.grid()

    plt.tight_layout()
    plt.show()

# 🎯 Main Function to Run the Sorting Benchmark
def main():
    input_sizes = [1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000,70000,100000]

    algorithms = {
        "1": ("Heap Sort", heap_sort),
        "2": ("Merge Sort", merge_sort),
        "3": ("Quick Sort", quick_sort),
        "4": ("Bucket Sort (Quick Sort)", bucket_sort),
        "5": ("Bucket Sort (Timsort)", bucket_sort_timsort),
        "6": ("Bucket Sort (Insertion)", bucket_sort_insertion),# New addition
        "7": ("Timsort", timsort),  # New addition
    }

    data_types = {
        "1": "random",
        "2": "sorted",
        "3": "reversed",
        "4": "large_duplicates",
        "5": "float",
    }

    print("Choose a sorting algorithm:")
    for key, (name, _) in algorithms.items():
        print(f"{key}. {name}")

    algo_choice = input("Enter the number of the sorting algorithm to use: ").strip()

    print("\nChoose the type of input data:")
    for key, name in data_types.items():
        print(f"{key}. {name.capitalize()}")

    data_choice = input("Enter the number of the input data type to use: ").strip()

    if algo_choice in algorithms and data_choice in data_types:
        algorithm_name, algorithm_function = algorithms[algo_choice]
        data_type = data_types[data_choice]

        print(f"\nRunning {algorithm_name} on {data_type} data (10 runs per size, using median values)...\n")
        time_results, memory_results = measure_performance(algorithm_function, input_sizes, data_type)
        plot_performance(input_sizes, time_results, memory_results, algorithm_name, data_type)
    else:
        print("Invalid choice. Please restart and choose a valid option.")

if __name__ == "__main__":
    main()
