import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class KalmanFilter:

    def __init__(self, F, Q, H, R):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.I = np.eye(2)

    def update(self, x, P, Z):
        y = Z - np.dot(self.H, x)
        S = np.dot(np.dot(self.H, P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(P, np.transpose(self.H)), np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = np.dot((self.I - np.dot(K, self.H)), P)
        return x, P

    def predict(self, x, P, u):
        x = self.F @ x + u
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def visualize(self, world_dim, mean, cov):
        mean = mean.reshape(-1)
        rv = multivariate_normal(mean, cov)
        x, y = np.mgrid[0:world_dim[0]:.5, 0:world_dim[1]:.5]
        pos = np.dstack((x, y))
        plt.contourf(x, y, rv.pdf(pos))
        plt.pause(0.05)
        plt.cla()

class Robot:

    def __init__(self, world_dim, motion_sigma=0, measurement_sigma=0):
        self.world_dim = world_dim
        self.x = np.random.random() * world_dim[0]
        self.y = np.random.random() * world_dim[1]
        self.theta = np.random.random() * 2 * np.pi
        self.motion_sigma = motion_sigma
        self.measurement_sigma = measurement_sigma

    def move(self, motion, rotation):
        self.theta += rotation
        self.theta %= 2 * np.pi
        dx = np.cos(self.theta) * motion + np.random.normal(0, self.motion_sigma)
        dy = np.sin(self.theta) * motion + np.random.normal(0, self.motion_sigma)
        self.x += dx
        self.y += dy

    def sense(self):
        x = self.x + np.random.normal(0, self.measurement_sigma)
        y = self.y + np.random.normal(0, self.measurement_sigma)
        return np.array([[x, y]]).T

    def random_action(self):
        motion = np.random.random() * 25
        rotation = np.random.random() * np.pi
        return motion, rotation

    def polar2cartesian(self, r, theta):
        theta += self.theta
        dx = np.cos(theta) * r
        dy = np.sin(theta) * r
        return dx, dy


world_dim = (100, 100)
motion_sigma = 2
measurement_sigma = 10
num_iterations = 100

robot = Robot(world_dim, motion_sigma, measurement_sigma)

F = np.eye(2)
Q = np.eye(2) * motion_sigma
H = np.eye(2)
R = np.eye(2) * measurement_sigma

kf = KalmanFilter(F, Q, H, R)

x = np.array([[0, 0]]).T
P = np.array([[1000, 0], [0, 1000]])

plt.show(block=False)

for _ in range(num_iterations):

    kf.visualize(world_dim, x, P)

    Z = robot.sense()
    x, P = kf.update(x, P, Z)
    r, theta = robot.random_action()
    robot.move(r, theta)
    cartesian = robot.polar2cartesian(r, theta)
    u = np.array([cartesian]).T
    x, P = kf.predict(x, P, u)

plt.close()






