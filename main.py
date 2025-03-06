import time
import random
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc


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


# 4️⃣ Bucket Sort (Using Quick Sort)
def bucket_sort_quick(arr):
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


# 5️⃣ Bucket Sort (Using Custom Timsort)
RUN_SIZE = 32


def insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


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

    return [num for bucket in buckets for num in bucket]


# 🚀 Sorting Algorithms Dictionary
algorithms = {
    "HeapSort": heap_sort,
    "MergeSort": merge_sort,
    "QuickSort": quick_sort,
    "BucketSort (QuickSort)": bucket_sort_quick,
    "BucketSort (Timsort)": bucket_sort_timsort,
    "Timsort": timsort,
}


# Function to measure execution time separately
def measure_execution_time(sort_func, arr, iterations=1):
    times = []
    for _ in range(iterations):
        copied_arr = arr[:]
        start_time = time.time()
        sort_func(copied_arr)
        execution_time = time.time() - start_time
        times.append(execution_time)
    return np.mean(times)


# Function to measure memory usage separately
def measure_memory_usage(sort_func, arr, iterations=5):
    memory_usages = []
    for _ in range(iterations):
        copied_arr = arr[:]
        tracemalloc.start()
        sort_func(copied_arr)
        memory_usage = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        memory_usages.append(memory_usage)
    return np.mean(memory_usages)


# Generating input data
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


# User selects input type
cases = ["random", "sorted", "reversed", "large_duplicates", "float"]
print("Choose an input type:")
for i, case in enumerate(cases, 1):
    print(f"{i}. {case.capitalize()}")

case_choice = input("Enter the number of the input type to use: ").strip()
if case_choice not in map(str, range(1, len(cases) + 1)):
    print("Invalid choice. Exiting.")
    exit()

selected_case = cases[int(case_choice) - 1]

# Running Benchmark
sizes = [1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 70000, 100000]
results = {}

input_datasets = {size: generate_data(size, selected_case) for size in sizes}
for name in algorithms:
    results.setdefault(name, {'time': [], 'memory': []})
    for size in sizes:
        time_taken = measure_execution_time(algorithms[name], input_datasets[size])
        memory_used = measure_memory_usage(algorithms[name], input_datasets[size])
        results[name]['time'].append(time_taken)
        results[name]['memory'].append(memory_used)

# Plot results
plt.figure(figsize=(10, 6))
for name in algorithms:
    plt.plot(sizes, results[name]['time'], label=name)
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title(f'Sorting Algorithm Performance - {selected_case.capitalize()} case')
plt.legend()
plt.grid()
plt.show()
