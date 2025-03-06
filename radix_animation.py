import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.animation as animation
import random


# 🔹 Heap Sort (Global Perspective)
def heap_sort(arr, frames):
    n = len(arr)

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
            frames.append((arr.copy(), {i, largest}))  # Capture full array
            heapify(arr, n, largest)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        frames.append((arr.copy(), {0, i}))  # Capture full array
        heapify(arr, i, 0)


# 🔹 Merge Sort (Global Perspective)
def merge_sort(arr, frames, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, frames, left, mid)
        merge_sort(arr, frames, mid + 1, right)
        merge(arr, left, mid, right, frames)


def merge(arr, left, mid, right, frames):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]
    changed_indices = set()

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        changed_indices.add(k)
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        changed_indices.add(k)
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        changed_indices.add(k)
        j += 1
        k += 1

    frames.append((arr.copy(), changed_indices))  # Capture full array


# 🔹 Quick Sort (Global Perspective)
def quick_sort(arr, frames, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left < right:
        pivot_index = partition(arr, left, right, frames)
        quick_sort(arr, frames, left, pivot_index - 1)
        quick_sort(arr, frames, pivot_index + 1, right)


def partition(arr, left, right, frames):
    pivot = arr[right]
    i = left - 1
    changed_indices = set()

    for j in range(left, right):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            changed_indices.add(i)
            changed_indices.add(j)

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    changed_indices.add(i + 1)
    changed_indices.add(right)

    frames.append((arr.copy(), changed_indices))  # Capture full array

    return i + 1


# 🔹 Bucket Sort (Using QuickSort, Fixed Visualization)
def bucket_sort_quick(arr, frames):
    if len(arr) == 0:
        return arr

    min_value, max_value = min(arr), max(arr)
    bucket_count = len(arr) // 10 or 1
    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = int((num - min_value) * (bucket_count - 1) // (max_value - min_value + 1))
        buckets[index].append(num)

    sorted_arr = []
    changed_indices = set()

    for i, bucket in enumerate(buckets):
        quick_sort(bucket, frames)  # Sort inside the bucket
        sorted_arr.extend(bucket)
        changed_indices.update(range(len(sorted_arr) - len(bucket), len(sorted_arr)))  # Track changes

    frames.append((sorted_arr.copy(), changed_indices))  # Capture full array
    return sorted_arr


# 🔹 Bucket Sort (Using Timsort, Fixed Visualization)
RUN_SIZE = 32


def insertion_sort(arr, left, right, frames):
    changed_indices = set()

    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            changed_indices.add(j + 1)
            j -= 1
        arr[j + 1] = key
        changed_indices.add(j + 1)
        frames.append((arr.copy(), changed_indices))


def timsort(arr, frames):
    n = len(arr)

    for i in range(0, n, RUN_SIZE):
        insertion_sort(arr, i, min(i + RUN_SIZE - 1, n - 1), frames)

    size = RUN_SIZE
    while size < n:
        for left in range(0, n, 2 * size):
            mid = left + size - 1
            right = min(left + 2 * size - 1, n - 1)
            if mid < right:
                merge_sort(arr, frames, left, right)
        size *= 2

    frames.append((arr.copy(), set(range(len(arr)))))  # Capture full array
    return arr


def bucket_sort_timsort(arr, frames):
    if len(arr) == 0:
        return arr

    min_value, max_value = min(arr), max(arr)
    bucket_count = len(arr) // 10 or 1
    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = int((num - min_value) * (bucket_count - 1) // (max_value - min_value + 1))
        buckets[index].append(num)

    sorted_arr = []
    changed_indices = set()

    for i, bucket in enumerate(buckets):
        timsort(bucket, frames)  # Sort inside the bucket
        sorted_arr.extend(bucket)
        changed_indices.update(range(len(sorted_arr) - len(bucket), len(sorted_arr)))  # Track changes

    frames.append((sorted_arr.copy(), changed_indices))  # Capture full array
    return sorted_arr


# 🚀 Sorting Algorithms Dictionary
algorithms = {
    "HeapSort": heap_sort,
    "MergeSort": merge_sort,
    "QuickSort": quick_sort,
    "BucketSort (QuickSort)": bucket_sort_quick,
    "BucketSort (Timsort)": bucket_sort_timsort,
    "Timsort": timsort,
}


def save_animation_as_gif(arr, sorting_function, filename="sorting_visualization.gif"):
    """Creates and saves an animated GIF of the sorting process"""
    fig, ax = plt.subplots()
    ax.set_title("Sorting Visualization")

    frames = []
    sorting_function(arr.copy(), frames)

    def update(frame_index):
        ax.clear()
        current_arr, changed_indices = frames[min(frame_index, len(frames) - 1)]

        # Ensure only modified bars are red
        colors = ['red' if i in changed_indices else 'blue' for i in range(len(current_arr))]

        ax.bar(range(len(current_arr)), current_arr, color=colors)
        ax.set_ylim(0, max(arr) + 1)
        ax.set_title(f"Sorting Step {frame_index + 1}")

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save(filename, writer="pillow", fps=10)
    print(f"Animation saved as {filename}")
    plt.close(fig)

# 🔹 Choose Sorting Algorithm
print("\nChoose a sorting algorithm:")
for i, (name, _) in enumerate(algorithms.items(), 1):
    print(f"{i}. {name}")

algo_choice = input("Enter the number of the sorting algorithm to use: ").strip()
selected_algorithm_name = list(algorithms.keys())[int(algo_choice) - 1]
selected_algorithm = algorithms[selected_algorithm_name]

# 🔹 Choose Array Size
array_size = int(input("Enter the number of elements to sort: ").strip())

# 🔹 Generate Random Array
arr = np.random.randint(10, 1000, array_size)

# 🔹 Run Sorting Visualization
save_animation_as_gif(arr, selected_algorithm, filename=f"{selected_algorithm_name}_visual.gif")
