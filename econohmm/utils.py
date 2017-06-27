__author__ = 'keithblackwell1'
def normalize(vec):
    """

    :rtype : object
    """
    z = vec.sum()
    return vec / z, z
