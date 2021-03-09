import numpy as np
from numpy.linalg import norm
import utils
from sklearn.decomposition import PCA
from findMin import findMin
from sklearn.metrics.pairwise import euclidean_distances as euc_dist

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        # D = utils.euclidean_dist_squared(X,X)
        # D = np.linalg.norm(X-X)**2
        D = euc_dist(X, X, squared=True)
        print(D)
        D = np.sqrt(D)
        print(D)

        # Initialize low-dimensional representation with PCA
        pca = PCA(k)
        pca.fit(X)
        Z = pca.transform(X)

        # Solve for the minimizer
        z, f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()



class ISOMAP(MDS):

    def __init__(self, n_components, n_neighbours):
        self.k = n_components
        self.nn = n_neighbours

    def compress(self, X):
        n = X.shape[0]

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # TODO: Convert these Euclidean distances into geodesic distances
        D_argsort = np.argsort(D)
        G = np.zeros(D.shape)
        for i in range(n):
            for nn in range(self.nn + 1):
                G[i, D_argsort[i, nn]] = D[i, D_argsort[i, nn]]
                G[D_argsort[i, nn], i] = D[D_argsort[i, nn], i] 

        # Compute shortest distance between points
        D = utils.dijkstra(G)

        # If two points are disconnected (distance is Inf)
        # then set their distance to the maximum
        # distance in the graph, to encourage them to be far apart.
        D[np.isinf(D)] = D[~np.isinf(D)].max()

        # Initialize low-dimensional representation with PCA
        pca = PCA(self.k)
        pca.fit(X)
        Z = pca.transform(X)

        # Solve for the minimizer
        z,f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, self.k)
        return Z
