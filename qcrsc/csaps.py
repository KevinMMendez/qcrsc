import numpy as np
import pandas as pd
from scipy.sparse import spdiags


class CubicSmoothSpline:
    """
    Univariate data approximation using a cubic smoothing spline.
    This is an implementation of the Fortran SMOOTH from PGS.
    Values are identical to matlab CSAPS.
    """

    def __init__(self, p=-1):
        self.p = p

    def fit(self, x, y):
        if isinstance(x, pd.Series):
            x = np.array(x)

        if isinstance(y, pd.Series):
            y = np.array(y)

        if isinstance(x, list):
            x = np.array(x)

        if isinstance(y, list):
            y = np.array(y)

        if sorted(list(x)) != list(x):
            raise ValueError("x needs to be in ascending order.")

        n = len(x)
        w = np.ones((1, n))
        dx = np.diff(x)
        divdif = np.diff(y) / dx[:]
        output = {}
        output["breaks"] = x

        if n < 2:
            raise ValueError("Need at least 2 data points.")
        if n == 2:
            # Straight line if n = 2
            self.p = 1
            output["coefs"] = np.concatenate([divdif, [y[0]]])
        else:
            # Spline
            R = spdiags(
                np.vstack(
                    (dx[1: n - 1], (dx[1: n - 1] + dx[0: n - 2]) * 2, dx[0: n - 2])
                ),
                [-1, 0, 1],
                n - 2,
                n - 2,
            )

            odx = 1 / dx
            Qt = spdiags(
                np.vstack(
                    (odx[0: n - 2], -(odx[1: n - 1] + odx[0: n - 2]), odx[1: n - 1])
                ),
                [0, -1, -2],
                n,
                n - 2,
            )

            # Solve for the 2nd derivates
            W = spdiags(w, 0, n, n)
            Qtw = Qt

            if self.p < 0:
                QtWQ = np.matmul(Qtw.toarray().T, Qtw.toarray())
                self.p = 1 / (1 + (np.trace(R.toarray())) / (6 * np.trace(QtWQ)))
                a = 6 * (1 - self.p) * QtWQ + self.p * R
                b = np.diff(divdif)
                u = np.linalg.lstsq(a, b, rcond=None)[0]
            else:
                a = (6 * (1 - self.p) * (np.matmul(Qtw.toarray().T, Qtw.toarray()))) + self.p * R
                b = np.diff(divdif)
                u = np.linalg.lstsq(a, b, rcond=None)[0]

            c = np.diff([np.concatenate([[0], u, [0]])]) / dx
            d = np.diff([np.concatenate([[0], c[0], [0]])])
            y = y - ((6 * (1 - self.p)) * W * d.T).T[0]

            c3 = np.concatenate([[0], self.p * u, [0]])
            c2 = np.diff(y) / dx - dx * (2 * c3[0: n - 1] + c3[1:n])

            output["coefs"] = np.vstack(
                (np.diff(c3) / dx, 3 * c3[0: n - 1], c2, y[0: n - 1])
            ).T

        self.output = output
        return self

    def predict(self, x):
        # Put into bins
        newoutput = self.output["breaks"][1:-1]

        if len(newoutput) == 0:
            # Linear regression as there are two data points
            newx = x - self.output["breaks"][0]
            c = self.output["coefs"][1]
            m = self.output["coefs"][0]
            val = m * newx + c
        else:
            idx = np.digitize(x, newoutput, right=False)

            # get initial newx
            newx = x - self.output["breaks"][idx]

            # nested multiplication
            val = self.output["coefs"][idx, 0]
            for i in range(1, len(self.output["coefs"].T)):
                val = val * newx + self.output["coefs"][idx, i]

        return val
