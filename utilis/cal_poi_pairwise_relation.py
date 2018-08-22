import dataset
import numpy as np
import scipy
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix


def cal_place_pairwise_dist(place_coordinates):
    # this method calculates the pair-wise rbf distance
    # returns a dict that stores the nearest neighbors of a place within a range of distance
    gamma = 60
    # place_neighbors_ind = get_nearest_places(location_coordinates)
    place_correlation = rbf_kernel(place_coordinates, gamma=gamma)
    np.fill_diagonal(place_correlation, 0)
    place_correlation[place_correlation < 0.1] = 0
    place_correlation = csr_matrix(place_correlation)

    return place_correlation


def main():
    # try attention model
    train_matrix, test_set, place_coords = dataset.Foursquare().generate_data()
    place_correlation = cal_place_pairwise_dist(place_coords)
    scipy.sparse.save_npz('./data/Foursquare/place_correlation_gamma60.npz', place_correlation)


if __name__ == '__main__':
    main()