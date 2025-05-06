import numpy as np
import os
from glob import glob

# Perform min-max scaling to ensure all values are in [-1, 1]
def rescale(data):
    scaled_data = data.copy()
    for sample in range(data.shape[0]):
        for src in range(data.shape[3]):
            for dim in range(data.shape[2]):
                col = data[sample, :, dim, src]
                min_val = np.min(col)
                max_val = np.max(col)

                # Avoid division by zero
                if max_val == min_val:
                    scaled_data[sample, :, dim, src] = 0.0
                else:
                    scaled_data[sample, :, dim, src] = 2 * (col - min_val) / (max_val - min_val) - 1
    return scaled_data

data_dir = './data' 

domain_names = ['T', 'RA', 'LA', 'RL', 'LL']
domain_indices = {
    'T': list(range(0, 9)),
    'RA': list(range(9, 18)),
    'LA': list(range(18, 27)),
    'RL': list(range(27, 36)),
    'LL': list(range(36, 45)),
}

train_x, train_y = [], []
test_x, test_y = [], []

train_x_norm, test_x_norm = [], []


for activity_idx in range(1, 20):  # a01–a19
    activity_folder = f"a{activity_idx:02d}"
    for person_idx in range(1, 9):  # p1–p8
        person_folder = f"p{person_idx}"
        segment_path = os.path.join(data_dir, activity_folder, person_folder, "*.txt")
        segment_files = sorted(glob(segment_path))

        for filepath in segment_files:

            try:
                data = np.loadtxt(filepath, delimiter=",")  # shape: (125, 45)
                if data.shape != (125, 45):
                    continue

                # Extract each domain (125, 9), then stack → (125, 9, 5)
                domain_stack = np.stack(
                    [data[:, domain_indices[d]] for d in domain_names], axis=-1
                )


                # Reshape to (125*9, 5) = (1125, 5)
                reshaped = domain_stack.reshape(125 * 9, 5)

                if person_idx <= 6:
                    train_x.append(reshaped)
                    train_x_norm.append(domain_stack)
                    train_y.append(activity_idx - 1)
                else:
                    test_x.append(reshaped)
                    test_x_norm.append(domain_stack)
                    test_y.append(activity_idx - 1)

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

# Convert to numpy arrays
train_x = np.array(train_x)  # shape: (6840, 1125, 5)
train_y = np.array(train_y)  # shape: (6840,)
test_x = np.array(test_x)    # shape: (2280, 1125, 5)
test_y = np.array(test_y)    # shape: (2280,)

# Normalize/rescale data
train_x_norm = rescale(np.array(train_x_norm))
train_x_norm = train_x_norm.reshape(train_x_norm.shape[0], train_x_norm.shape[1] * train_x_norm.shape[2], train_x_norm.shape[3])
test_x_norm = rescale(np.array(test_x_norm))
test_x_norm = test_x_norm.reshape(test_x_norm.shape[0], test_x_norm.shape[1] * test_x_norm.shape[2], test_x_norm.shape[3])

np.save("data_train.npy", train_x)
np.save("data_test.npy", test_x)
np.save("target_train.npy", train_y)
np.save("target_test.npy", test_y)
np.save("data_train_norm.npy", train_x_norm)
np.save("data_test_norm.npy", test_x_norm)