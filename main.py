import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def initialize_clusters(points, k):
    res = points[np.random.randint(points.shape[0], size=k)]
    return res


def get_distances(centroid, points):
    res = np.linalg.norm(points - centroid, axis=1)
    return res


def kmeans_objective(X, classes, centroids):
    obj = 0
    for i, c in enumerate(centroids):
        obj += np.sum(np.linalg.norm(X[classes == i] - c, axis=1) ** 2)
    return obj


def gaussian(x, mean, cov):
    d = len(x)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    coefficient = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(cov)))
    return coefficient * np.exp(exponent)


def calculateResponsibilities(x, weights, means, covariances):
    responsibilities = []
    for weight, mean, cov in zip(weights, means, covariances):
        responsibilities.append(weight * gaussian(x, mean, cov))
    total_weighted_likelihood = np.sum(responsibilities)
    responsibilities /= total_weighted_likelihood
    return responsibilities


def learning(X, k=3, maxiter=50):
    centroids = initialize_clusters(X, k)

    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)

    for i in range(maxiter):

        for j, c in enumerate(centroids):
            distances[:, j] = get_distances(c, X)

        classes = np.argmin(distances, axis=1)

        for c in range(k):
            centroids[c] = np.mean(X[classes == c], 0)

        if i < 5:
            plt.figure()
            group_colors = ['skyblue', 'coral']
            colors = [group_colors[j] for j in classes]
            plt.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
            plt.scatter(centroids[:, 0], centroids[:, 1], color=['blue', 'darkred'], marker='o', lw=2)
            plt.title("Iteration {}".format(i + 1))
            plt.show()

            print("Iteration {}".format(i + 1), "Objective Function Value: ", kmeans_objective(X, classes, centroids))

    return classes, centroids


def PartOne():
    np.random.seed(0)
    X1 = np.random.multivariate_normal([2, 1], np.diag([0.4, 0.04]), 100)
    X2 = np.random.multivariate_normal([1, 2], np.diag([0.4, 0.04]), 100)
    X = np.concatenate((X1, X2), axis=0)

    classes, centroids = learning(X, 2)
    objective = kmeans_objective(X, classes, centroids)

    group_colors = ['skyblue', 'coral']
    colors = [group_colors[j] for j in classes]

    plt.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], color=['blue', 'darkred'], marker='o', lw=2)
    plt.show()

    print("K-Means Objective Function Value:", objective)


def PartTwo():
    np.random.seed(0)
    X1 = np.random.multivariate_normal([2, 1], np.diag([0.4, 0.04]), 100)
    X2 = np.random.multivariate_normal([1, 2], np.diag([0.4, 0.04]), 100)
    X = np.concatenate((X1, X2), axis=0)

    gm = GaussianMixture(n_components=2, random_state=0).fit(X)

    means = gm.means_
    prediction = gm.predict(X)

    group_colors = ['skyblue', 'coral']
    colors = [group_colors[j] for j in prediction]

    plt.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    plt.scatter(means[:, 0], means[:, 1], color=['blue', 'darkred'], marker='o', lw=2)
    plt.title("Gaussian Mixture Model Clustering")
    plt.show()

    print("Weights:")
    print(gm.weights_)
    print("\nMeans:")
    print(gm.means_)
    print("\nCovariance Matrices:")
    print(gm.covariances_)

    point = np.array([1.5, 1.5])
    responsibilities = calculateResponsibilities(point, gm.weights_, gm.means_, gm.covariances_)
    print("Responsibility values:", responsibilities)

    assignedCluster = np.argmax(responsibilities)
    print("Cluster assignment using responsibility values:", assignedCluster)

    assignedCluster1 = gm.predict([point])[0]
    print("Cluster assignment using gm.predict:", assignedCluster1)


def main():
    print("PART ONE RESULTS:\n")
    PartOne()
    print("\nPART TWO RESULTS:\n")
    PartTwo()


main()
