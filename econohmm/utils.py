def normalize(vec):
    """

    :rtype : object
    """
    z = vec.sum()
    return vec / z, z
