import time
import random
import tracemalloc
import numpy as np


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


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


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
                merge_sort(arr[left:right + 1])
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


algorithms = {
    "HeapSort": heap_sort,
    "MergeSort": merge_sort,
    "QuickSort": quick_sort,
    "BucketSort (QuickSort)": bucket_sort_quick,
    "BucketSort (Timsort)": bucket_sort_timsort,
    "Timsort": timsort,
}


def measure_execution_time(sort_func, arr):
    copied_arr = arr[:]
    start_time = time.time()
    sort_func(copied_arr)
    return time.time() - start_time


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


def choose_best_algorithm(arr, case):
    size = len(arr)

    if size >= 500000:
        return "BucketSort (Timsort)"

    if size <= 10000:
        return "QuickSort"

    elif arr == sorted(arr):
        return "Timsort"
    elif arr == sorted(arr, reverse=True):
        return "HeapSort"

    elif len(set(arr)) < size / 10:
        return "BucketSort (QuickSort)"

    return "BucketSort (Timsort)"



data_types = ["random", "sorted", "reversed", "large_duplicates", "float"]
selected_case = random.choice(data_types)
array_size = random.choice([10000])

print(f"\n🔹 Auto-Generated Input Type: {selected_case.capitalize()}")
print(f"🔹 Auto-Generated Array Size: {array_size}\n")


arr = generate_data(array_size, selected_case)

best_algorithm = choose_best_algorithm(arr, selected_case)

start_time = time.time()
sorted_arr = algorithms[best_algorithm](arr)
execution_time = time.time() - start_time

print(f"✅ Automatically Selected Algorithm: {best_algorithm}")
print(f"⏱ Execution Time: {execution_time:.6f} seconds")
print("✔ Sorting Complete!")
