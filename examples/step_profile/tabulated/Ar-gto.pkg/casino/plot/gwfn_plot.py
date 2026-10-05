#!/usr/bin/env python3

import argparse
import numpy as np

from plot_implicit import plot_implicit
from backflow import Backflow
from gwfn import Gwfn


def Be_plot(wfn, bf=None):

    def plot(r12_minus, r34_minus, r12_plus):
        """
        """
        # r12_plus = r12_plus + 3.0
        r34_plus = np.full((100, 100), 1.0)
        vec_1 = vec_3 = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
        phi = np.pi/4.0
        theta = np.pi/2.0
        vec_2 = vec_4 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])[:, np.newaxis, np.newaxis]
        r1I = (r12_plus + r12_minus)/2.0
        r2I = (r12_plus - r12_minus)/2.0
        r1 = vec_1 * r1I[np.newaxis]
        r2 = vec_2 * r2I[np.newaxis]
        r3I = (r34_plus + r34_minus)/2.0
        r4I = (r34_plus - r34_minus)/2.0
        r3 = vec_3 * r3I[np.newaxis]
        r4 = vec_4 * r4I[np.newaxis]
        if bf:
            return wfn(*bf(r1, r2, r3, r4))
        else:
            return wfn(r1, r2, r3, r4)

    return plot


def main():
    parser = argparse.ArgumentParser(
        description="This script to read gwfn.data file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'gwfn_file',
        type=str,
        default='gwfn.data',
        nargs='?',
        help="name of gwfn.data file"
    )

    parser.add_argument(
        'backflow_file',
        type=str,
        default='correlation.data',
        nargs='?',
        help="name of correlation.data file"
    )

    args = parser.parse_args()

    Be_gwfn = Gwfn(args.gwfn_file)
    Be_backflow = Backflow(args.backflow_file)

    if False:
        fn = Be_plot(Be_gwfn.Be_1s2s, Be_backflow.Be)
    else:
        fn = Be_plot(Be_gwfn.Be_4det)

    plot_implicit(fn)


if __name__ == '__main__':
    main()
