import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class Particle:

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class ParticleFilter:

    def __init__(self, world_dim, num_particles=5000, dist_sigma=0.05, theta_sigma=0.05, sense_sigma=5):
        self.world_dim = world_dim
        self.num_particles = num_particles
        self.dist_sigma = dist_sigma
        self.theta_sigma = theta_sigma
        self.sense_sigma = sense_sigma
        self.particles = self.generate_random_particles()

    def generate_random_particles(self):
        particles = []
        for _ in range(self.num_particles):
            x = np.random.random() * self.world_dim[0]
            y = np.random.random() * self.world_dim[1]
            theta = np.random.random() * 2 * np.pi
            particles.append(Particle(x, y, theta))
        return particles
        
    def motion_update(self, delta_dist, delta_theta):
        particles = []
        delta_dist += np.random.normal(0, self.dist_sigma)
        delta_theta += np.random.normal(0, self.theta_sigma)
        for p in self.particles:
            theta = p.theta + delta_theta
            theta %= 2 * np.pi
            x = p.x + delta_dist * np.cos(theta)
            x %= self.world_dim[0]
            y = p.y + delta_dist * np.sin(theta)
            y %= self.world_dim[1]
            particles.append(Particle(x, y, theta))
        self.particles = particles

    def measurement_update(self, Z, landmarks):
        weights = []
        for particle in self.particles:
            prob = 1
            for i, (lx, ly) in enumerate(landmarks):
                dist = np.sqrt((particle.x - lx) ** 2 + (particle.y - ly) ** 2)
                prob *= norm.pdf(Z[i], dist, self.sense_sigma)
            weights.append(prob)
        weights = np.array(weights)
        weights /= np.sum(weights)
        self.particles = np.random.choice(self.particles, size=self.num_particles, p=weights)

class Robot:

    def __init__(self, world_dim, dist_sigma=0, theta_sigma=0, sense_sigma=0):
        self.world_dim = world_dim
        self.x = np.random.random() * world_dim[0]
        self.y = np.random.random() * world_dim[1]
        self.theta = np.random.random() * 2 * np.pi
        self.dist_sigma = dist_sigma
        self.theta_sigma = theta_sigma
        self.sense_sigma = sense_sigma

    def move(self, delta_dist, delta_theta):
        delta_dist += np.random.normal(0, self.dist_sigma)
        delta_theta += np.random.normal(0, self.theta_sigma)
        self.theta += delta_theta
        self.theta %= 2 * np.pi
        self.x += delta_dist * np.cos(self.theta)
        self.x %= self.world_dim[0]
        self.y += delta_dist * np.sin(self.theta)
        self.y %= self.world_dim[1]

    def sense(self, landmarks):
        Z = []
        for lx, ly in landmarks:
            dist = np.sqrt((self.x - lx) ** 2 + (self.y - ly) ** 2)
            dist += np.random.normal(0, self.sense_sigma)
            Z.append(dist)
        return Z


def render(world_dim, robot, particle_filter, landmarks):
    plt.xlim(0, world_dim[0])
    plt.ylim(0, world_dim[1])
    plt.scatter([p.x for p in particle_filter.particles], 
        [p.y for p in particle_filter.particles], color='r')
    plt.scatter([x for x, _ in landmarks], [y for _, y in landmarks], color='g')
    plt.scatter([robot.x], [robot.y], color='b')
    plt.pause(0.05)
    plt.cla()

def simulate(world_dim, robot, particle_filter, landmarks):
    render(world_dim, robot, pf, landmarks)
    dist = np.random.random() * 10
    theta = np.random.random() * np.pi
    robot.move(dist, theta)
    Z = robot.sense(landmarks)
    particle_filter.motion_update(dist, theta)
    particle_filter.measurement_update(Z, landmarks)
    

num_iterations = 10
world_dim = (100, 100)
landmarks  = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]

robot = Robot(world_dim)
pf = ParticleFilter(world_dim)

plt.show(block=False)
for i in range(num_iterations):
    simulate(world_dim, robot, pf, landmarks)
plt.close()


