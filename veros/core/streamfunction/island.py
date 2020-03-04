import np as np
import scipy.ndimage

from veros.core import veros_kernel
from .. import utilities


@veros_kernel(static_args=('enable_cyclic_x',))
def isleperim(kmt, enable_cyclic_x, verbose=False):
    utilities.enforce_boundaries(kmt, enable_cyclic_x)

    structure = np.ones((3, 3))  # merge diagonally connected land masses

    # find all land masses
    labelled, _ = scipy.ndimage.label(kmt == 0, structure=structure)

    # find and set perimeter
    land_masses = labelled > 0
    inner = scipy.ndimage.binary_dilation(land_masses, structure=structure)
    perimeter = np.logical_xor(inner, land_masses)
    labelled[perimeter] = -1

    # match wrapping periodic land masses
    if enable_cyclic_x:
        west_slice = labelled[2]
        east_slice = labelled[-2]

        for west_label in np.unique(west_slice[west_slice > 0]):
            east_labels = np.unique(east_slice[west_slice == west_label])
            east_labels = east_labels[~np.isin(east_labels, [west_label, -1])]
            if not east_labels.size:
                # already labelled correctly
                continue
            assert len(np.unique(east_labels)) == 1, (west_label, east_labels)
            labelled[labelled == east_labels[0]] = west_label

    utilities.enforce_boundaries(labelled, enable_cyclic_x)

    # label landmasses in a way that is consistent with pyom
    labels = np.unique(labelled[labelled > 0])

    label_idx = {}
    for label in labels:
        # find index of first island cell, scanning west to east, north to south
        label_idx[label] = np.argmax(labelled[:, ::-1].T == label)

    sorted_labels = list(sorted(labels, key=lambda i: label_idx[i]))

    # ensure labels are numbered consecutively
    relabelled = labelled.copy()
    for new_label, label in enumerate(sorted_labels, 1):
        if label == new_label:
            continue
        relabelled[labelled == label] = new_label

    return np.asarray(relabelled)
