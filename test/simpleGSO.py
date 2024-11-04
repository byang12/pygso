from math import exp, cos, pi
from pygso.algorithm import GSOBuilder
from pygso.parameters import GSOParameters
from pygso.ofunction import ObjectiveFunction
from pygso.boundaries import BoundingBox, Boundary
from pygso.gso_random import MTGenerator

# Custom optimization function inherits from ObjectiveFunction class:
class Rastrigin(ObjectiveFunction):
    """Rastrigin function"""

    def __call__(self, coordinates):
        return Rastrigin.calculate(coordinates[0], coordinates[1])

    @staticmethod
    def calculate(x, y):
        return (
            20.0 + (x*x - 10.0*cos(2*pi*x) + y*y - 10.0*cos(2*pi*y))
        )

# New instance of the function to optimize
objective_function = Rastrigin()

# Limits of the function to be sampled, one instance of Boundary 
# for each dimension:
bounding_box = BoundingBox([Boundary(-3.0, 3.0), Boundary(-3.0, 3.0)])

# Number of glowworms of the simulation
number_of_glowworms = 200

# Random number generator for producing starting glowworm positions in the
# function landscape
seed = 324324
random_number_generator = MTGenerator(seed)

# Parameters of the algorithm (rho, beta, etc. with default values)
parameters = GSOParameters()

# Algorithm factory
builder = GSOBuilder() 
gso = builder.create(
    number_of_glowworms,
    random_number_generator,
    parameters,
    objective_function,
    bounding_box,
)

# Run this optimization for 50 steps
gso.run(50)

print(gso)