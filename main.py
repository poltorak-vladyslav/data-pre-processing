import numpy as np
from sklearn import preprocessing

def main():
    input_data = np.array([[2.3, 4.6, 3.2],
                          [5.1, 9.3, 5.9],
                          [2.8, 4.1, 7.5]])

    data_binarized = preprocessing.Binarizer(threshold=4.0).transform(input_data)
    print("\nБінаризовані дані:\n", data_binarized)

    print("\nДо:")
    print("Середнє значення:", input_data.mean(axis=0))
    print("Стандартне відхилення:", input_data.std(axis=0))

    data_scaled = preprocessing.scale(input_data)
    print("\nПісля:")
    print("Середнє значення:", data_scaled.mean(axis=0))
    print("Стандартне відхилення:", data_scaled.std(axis=0))

    data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
    print("\nМасштабовані MinMax дані:\n", data_scaled_minmax)

    data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
    data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
    print("\nL1-нормалізовані дані:\n", data_normalized_l1)
    print("\nL2-нормалізовані дані:\n", data_normalized_l2)

if __name__ == "__main__":
    main()
