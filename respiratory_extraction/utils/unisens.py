import unisens
import numpy as np


def read_unisens_entry(path: str, entry: str) -> tuple[np.ndarray, int]:
    """
    Get the data and sample rate from an unisens entry
    :param path:  path to the unisens directory
    :param entry: entry to read
    :return: data and sample rate
    """

    data_entry = unisens.Unisens(path, readonly=True)[entry]

    # The data is stored as a (1, X) array, but we want to flatten it to a 1D array
    raw_signal = data_entry.get_data().flatten()

    sample_rate = int(data_entry.attrib['sampleRate'])

    return raw_signal, sample_rate
