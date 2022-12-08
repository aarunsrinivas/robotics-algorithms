import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class KalmanFilter:

    def __init__(self, F, Q, H, R, dt=1):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.dt = dt
        self.I = np.eye(3)

    def B(self, x):
        gamma = x[-1, 0]
        return np.array([
            [np.cos(gamma) * self.dt, 0],
            [np.sin(gamma) * self.dt, 0],
            [0, self.dt]
        ])

    def update(self, x, P, Z):
        y = Z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.pinv(S)
        x = x + K @ y
        P = (self.I - K @ self.H) @ P
        return x, P

    def predict(self, x, P, u):
        x = self.F @ x + self.B(x) @ u
        P = self.F @ P @ self.F.T + self.Q
        return x, P


class Robot:

    def __init__(self, x, motion_sigma, measurement_sigma, dt=1):
        self.x = x.copy()
        self.motion_sigma = motion_sigma
        self.measurement_sigma = measurement_sigma
        self.dt = dt

    @property
    def B(self):
        gamma = self.x[-1, 0]
        return np.array([
            [np.cos(gamma) * self.dt, 0],
            [np.sin(gamma) * self.dt, 0],
            [0, self.dt]
        ])

    @property
    def motion_noise(self):
        return np.array([[np.random.normal(0, sigma) for sigma in self.motion_sigma]]).T

    @property
    def measurement_noise(self):
        return np.array([[np.random.normal(0, sigma) for sigma in self.measurement_sigma]]).T

    def move(self, u):
        self.x += self.B @ u
        self.x += self.motion_noise

    def sense(self):
        Z = self.x + self.measurement_noise
        return Z

    def random_action(self):
        v = np.random.random() * 5
        w = np.random.random() * np.pi
        return np.array([[v, 0]]).T


def visualize(robot, x, P, time=0.5):
    rv = multivariate_normal(x.reshape(-1)[:2], P[:2, :2])
    i, j = np.mgrid[-100:100:1, -100:100:1]
    points = np.dstack((i, j))
    plt.contourf(i, j, rv.pdf(points))

    pos = robot.x.reshape(-1)
    plt.scatter([pos[0]], [pos[1]], color='b')
    plt.pause(time)
    plt.cla()

dt = 1
motion_sigma = [10, 10, 0.2]
measurement_sigma = [3, 3, 0.1]

x = np.array([[0., 0., 0.]]).T
P = np.diag([1000., 1000., 1000.])

robot = Robot(x, motion_sigma, measurement_sigma, dt=dt)

F = np.eye(3)
Q = np.diag(motion_sigma)
H = np.eye(3)
R = np.diag(measurement_sigma)

kf = KalmanFilter(F, Q, H, R, dt=dt)

plt.show(block=False)

num_iterations = 10
for i in range(num_iterations):

    Z = robot.sense()
    x, P = kf.update(x, P, Z)
    
    u = robot.random_action()
    robot.move(u)
    x, P = kf.predict(x, P, u)

    visualize(robot, x, P, time=0.5)

plt.close()






