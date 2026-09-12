import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    
    q1, feature_dim = desc1.shape
    q2 = desc2.shape[0]

    distances = (desc1**2)@np.ones((feature_dim, q2)) - 2*(desc1@np.transpose(desc2)) + np.transpose((desc2**2)@np.ones((feature_dim, q1)))
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis

        matches = np.stack((np.arange(q1), np.argmin(distances, axis=1)), axis=1)
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis

        m1 = np.argmin(distances, axis=1)
        m2 = np.argmin(distances, axis=0)
        index = np.arange(q1) == m2[m1]
        matches = np.stack((np.arange(q1), m1), axis=1)[index]
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row

        m1 = np.argmin(distances, axis=1)
        m2 = np.partition(distances, 2, axis=1)[:,1]
        index = distances[range(distances.shape[0]), m1] / m2 < ratio_thresh
        matches = np.stack((np.arange(q1), m1), axis=1)[index] 
    else:
        raise NotImplementedError
    return matches

