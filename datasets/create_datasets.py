import pandas as pd
import numpy as np
#from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import make_s_curve

# Note some of these functions were created with help of ChatGPT 4.

# Blobs
def make_blobs_data(n_samples, n_features, n_centers, 
                    shuffle, random_state, centerx, centery):
    
    raw_df = make_blobs(
        n_samples = n_samples, 
        n_features = n_features, 
        centers = n_centers,
        shuffle = shuffle, 
        random_state=random_state,
        center_box=(centerx, centery)
    )

    features, labels = raw_df
    df = pd.DataFrame(features)
    df = df.add_prefix('x')
    df['y'] = labels
    return df

# def make_non_overlapping_moons(n_samples=5000, noise=0.1, random_state=None):
#     """
#     Generates a dataset with six non-overlapping sets of moons.
    
#     Parameters:
#     -----------
#     n_samples: int, optional (default=5000)
#         The total number of samples to generate.
#     noise: float, optional (default=0.1)
#         The standard deviation of the Gaussian noise added to the data.
#     random_state: int, RandomState instance or None, optional (default=None)
#         Determines random number generation for dataset creation. Pass an int
#         for reproducible output across multiple function calls.
        
#     Returns:
#     --------
#     A pandas DataFrame containing the generated data with three columns: 'x', 'y', and 'label'.
#     """
    
#     # Define the number of samples for each set of moons
#     n_samples_per_moon = n_samples // 6
    
#     # Define the x-coordinate shifts for the six sets of moons
#     shifts = [[-4, 0], [-2, -2], [0, -4], [2, -2], [4, 0], [2, 2]]
    
#     # Generate the six sets of moons with non-overlapping x-coordinates
#     X = np.empty((0, 2))
#     y = np.empty(0)
#     for i, shift in enumerate(shifts):
#         X_moon, y_moon = make_moons(n_samples=n_samples_per_moon, noise=noise, random_state=random_state+i)
#         X_moon[:, 0] += shift[0]
#         X_moon[:, 1] += shift[1]
#         X = np.concatenate([X, X_moon], axis=0)
#         y = np.concatenate([y, y_moon], axis=0)
    
#     # Create a dataframe from the X and y arrays
#     df = pd.DataFrame({'x0': X[:, 0], 'x1': X[:, 1], 'y': y})
    
#     return df

def make_non_overlapping_scurves(n_samples=5000, n_curves=6, noise=0.1, random_state=None):
    """
    Generates a dataset with a specified number of non-overlapping S-curve datasets and assigns a label to each dataset.

    Parameters:
    -----------
    n_samples: int, optional (default=5000)
        The total number of samples to generate.
    n_curves: int, optional (default=6)
        The number of S-curves to generate.
    noise: float, optional (default=0.1)
        The standard deviation of the Gaussian noise added to the data.
    random_state: int, RandomState instance or None, optional (default=None)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns:
    --------
    A pandas DataFrame containing the generated data with four columns: 'x0', 'x1', 'x2', and 'y'.
    """

    # Calculate the number of samples for each S-curve dataset
    base_n_samples_per_curve = n_samples // n_curves
    remaining_samples = n_samples % n_curves
    n_samples_per_curve_list = [base_n_samples_per_curve] * n_curves
    for i in range(remaining_samples):
        n_samples_per_curve_list[i] += 1

    # Define the x-coordinate shifts and labels for the S-curve datasets
    curves = [(np.array([i * 2, -i, 0]), i) for i in range(n_curves)]

    # Generate the non-overlapping S-curve datasets and assign labels
    X = np.empty((0, 3))
    y = np.empty((0,))
    for i, (shift, label) in enumerate(curves):
        n_samples_per_curve = n_samples_per_curve_list[i]
        X_curve, _ = make_s_curve(n_samples=n_samples_per_curve, noise=noise, random_state=random_state+i)
        X_curve[:, 0] += shift[0]
        X_curve[:, 1] += shift[1]
        X_curve[:, 2] += shift[2]
        X = np.concatenate([X, X_curve], axis=0)
        y_curve = np.full((n_samples_per_curve,), label)
        y = np.concatenate([y, y_curve], axis=0)

    # Create a dataframe from the X and y arrays
    df = pd.DataFrame({'x0': X[:, 0], 'x1': X[:, 1], 'x2': X[:, 2], 'y': y})

    return df

def main():
    print("Creating blobs dataset with 8 centers and 3 features...")
    # SET 1: center-based, N = 10000, 8 centers, 3 features
    make_blobs_data(n_samples = 10000, 
                n_features = 3, 
                n_centers = 8,
                shuffle = True, 
                random_state=42,
                centerx = -20.0, 
                centery = 20.0).to_csv('./datasets/blobs_c8_f3.csv', 
                                       index = False)
    print("Blobs dataset with 8 centers and 3 features created.")
    print()
    
    # SET 2: center-based, N = 10000, 20 centers, 3 features
    print("Creating blobs dataset with 19 centers and 3 features...")
    make_blobs_data(n_samples = 10000, 
                n_features = 3, 
                n_centers = 19,
                shuffle = True, 
                random_state=42,
                centerx = -20.0, 
                centery = 20.0).to_csv('./datasets/blobs_c19_f3.csv', 
                                       index = False)
    print("Blobs dataset with 19 centers and 3 features created.")
    print()

    # SET 3: center-based, N = 10000, 6 centers, 10 features
    print("Creating blobs dataset with 6 centers and 10 features...")
    make_blobs_data(n_samples = 10000, 
                n_features = 10, 
                n_centers = 6,
                shuffle = True, 
                random_state=42,
                centerx = -20.0, 
                centery = 20.0).to_csv('./datasets/blobs_c6_f10.csv', 
                                       index = False)
    print("Blobs dataset with 6 centers and 10 features created.")
    print()

    # SET 4: center-based, N = 10000, 19 centers, 10 features
    print("Creating blobs dataset with 19 centers and 10 features...")
    make_blobs_data(n_samples = 10000, 
                n_features = 10, 
                n_centers = 19,
                shuffle = True, 
                random_state=42,
                centerx = -20.0, 
                centery = 20.0).to_csv('./datasets/blobs_c19_f10.csv', 
                                       index = False)
    print("Blobs dataset with 19 centers and 19 features created.")
    print()

    ## SET 5: Non-centric S-curves, N = 10000, n_curves = 3, 8 features
    print("Creating S-curves dataset with 8 centers and 3 features...")
    make_non_overlapping_scurves(n_samples=5000, n_curves=8, 
                                 noise=0.1, random_state=42).sample(10000, 
                                                           replace = True, 
                                                           random_state=42).reset_index(drop = True).to_csv('./datasets/scurves_c8_f3.csv', index = False)
    print("S-curves dataset with 3 S-curves and 8 features created.")
    print() 

    ## SET 6: Non-centric S-curves, N = 10000, n_curves = 3, 3 features
    print("Creating S-curves dataset with 3 centers and 3 features...")
    make_non_overlapping_scurves(n_samples=5000, n_curves=3, 
                                 noise=0.1, random_state=42).sample(10000, 
                                                           replace = True, 
                                                           random_state=42).reset_index(drop = True).to_csv('./datasets/scurves_c3_f3.csv', index = False)
    print("S-curves dataset with 3 S-curves and 3 features created.")
    print()
    

if __name__ == '__main__':
    main()