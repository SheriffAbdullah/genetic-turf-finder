import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

# Constants
EARTH_RADIUS_KM = 6371
LAT_BOUNDS = (0, 90)
LON_BOUNDS = (-180, 180)

# Function Definitions

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def fitness_function(chromosome: Tuple[float, float], user_locations: List[Tuple[float, float]], 
                     weights: Optional[List[float]] = None) -> float:
    """
    Calculate the fitness of a chromosome based on distances to user locations.
    """
    lat, lon = chromosome
    distances = [haversine_distance(lat, lon, user_lat, user_lon) for user_lat, user_lon in user_locations]
    return np.sum(np.array(distances) * np.array(weights)) if weights else np.sum(distances)

def generate_initial_population(pop_size: int, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> np.ndarray:
    """
    Generate an initial population of chromosomes within specified bounds.
    """
    lat_bounds, lon_bounds = bounds
    return np.array([[np.random.uniform(*lat_bounds), np.random.uniform(*lon_bounds)] for _ in range(pop_size)])

def selection(population: np.ndarray, fitnesses: np.ndarray, num_parents: int) -> np.ndarray:
    """
    Select the best individuals from the population based on fitness.
    """
    selected_indices = np.argsort(fitnesses)[:num_parents]
    return population[selected_indices]

def crossover(parents: np.ndarray, offspring_size: Tuple[int, int]) -> np.ndarray:
    """
    Perform crossover to generate offspring.
    """
    offspring = np.empty(offspring_size)
    crossover_point = offspring_size[1] // 2
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover: np.ndarray, mutation_rate: float, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> np.ndarray:
    """
    Perform mutation on the offspring.
    """
    lat_bounds, lon_bounds = bounds
    for idx in range(offspring_crossover.shape[0]):
        if np.random.rand() < mutation_rate:
            offspring_crossover[idx, 0] = np.random.uniform(*lat_bounds)
        if np.random.rand() < mutation_rate:
            offspring_crossover[idx, 1] = np.random.uniform(*lon_bounds)
    return offspring_crossover

def visualize(user_locations: List[Tuple[float, float]], best_location: Tuple[float, float], 
              turfs: Optional[List[Tuple[float, float]]] = None) -> None:
    """
    Visualize user locations, turfs, and the best turf.
    """
    user_lats, user_lons = zip(*user_locations)
    plt.scatter(user_lons, user_lats, c='blue', label='Users')
    if turfs:
        turf_lats, turf_lons = zip(*turfs)
        plt.scatter(turf_lons, turf_lats, c='green', label='Available Turfs')
        for j, (turf_lon, turf_lat) in enumerate(zip(turf_lons, turf_lats)):
            plt.annotate(f'Turf {j}', (turf_lon, turf_lat), textcoords="offset points", xytext=(10, -10), ha='center', fontsize=8, color='green')
    plt.scatter(best_location[1], best_location[0], c='red', label='Best Turf', edgecolors='black', s=100, zorder=5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('User Locations and Optimal Turf')
    plt.legend()
    plt.show()

def detect_outliers(distances: List[float], threshold: float = 1.5) -> Tuple[List[int], List[int]]:
    """
    Detect outliers based on the threshold applied to distance statistics.
    """
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    outlier_limit = mean_distance + threshold * std_distance
    non_outliers = [i for i, d in enumerate(distances) if d <= outlier_limit]
    outliers = [i for i, d in enumerate(distances) if d > outlier_limit]
    return non_outliers, outliers

def genetic_algorithm(user_locations: List[Tuple[float, float]], pop_size: int = 100, num_generations: int = 100, 
                      mutation_rate: float = 0.01, bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (LAT_BOUNDS, LON_BOUNDS),
                      weights: Optional[List[float]] = None, turfs: Optional[List[Tuple[float, float]]] = None) -> Tuple[float, float]:
    """
    Genetic algorithm to find the optimal turf.
    """
    if turfs:
        population = np.array(turfs)
        fitnesses = np.array([fitness_function(chromosome, user_locations, weights) for chromosome in population])
        best_idx = np.argmin(fitnesses)
        best_location = population[best_idx]
        visualize(user_locations, best_location, turfs)
        return best_location

    population = generate_initial_population(pop_size, bounds)
    for _ in range(num_generations):
        fitnesses = np.array([fitness_function(chromosome, user_locations, weights) for chromosome in population])
        parents = selection(population, fitnesses, pop_size // 2)
        offspring_crossover = crossover(parents, offspring_size=(pop_size - parents.shape[0], 2))
        offspring_mutation = mutation(offspring_crossover, mutation_rate, bounds)
        population[parents.shape[0]:] = offspring_mutation
    best_idx = np.argmin([fitness_function(chromosome, user_locations, weights) for chromosome in population])
    best_location = population[best_idx]
    visualize(user_locations, best_location)
    return best_location

def genetic_algorithm_with_outlier_discounting(
    user_locations: List[Tuple[float, float]],
    pop_size: int = 100,
    num_generations: int = 100,
    mutation_rate: float = 0.01,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (LAT_BOUNDS, LON_BOUNDS),
    weights: Optional[List[float]] = None,
    turfs: Optional[List[Tuple[float, float]]] = None,
    enable_outlier_discounting: bool = False,
    outlier_threshold: float = 1.5
) -> Dict[str, any]:
    """
    Genetic algorithm with optional outlier discounting.
    """
    best_location = genetic_algorithm(user_locations, pop_size, num_generations, mutation_rate, bounds, weights, turfs)
    distances = [haversine_distance(user[0], user[1], best_location[0], best_location[1]) for user in user_locations]

    if enable_outlier_discounting:
        non_outlier_indices, outlier_indices = detect_outliers(distances, threshold=outlier_threshold)
        non_outliers = [user_locations[i] for i in non_outlier_indices]
        outliers = [user_locations[i] for i in outlier_indices]

        new_best_location = genetic_algorithm(non_outliers, pop_size, num_generations, mutation_rate, bounds, weights, turfs)

        return {
            "initial_best_turf": best_location,
            "new_best_turf": new_best_location,
            "outliers": outliers,
            "non_outliers": non_outliers
        }
    
    return {
        "initial_best_turf": best_location,
        "outliers": [],
        "non_outliers": user_locations
    }

def find_optimal_turf(user_locations: List[Tuple[float, float]], turfs: List[Tuple[float, float]], 
                      priority_user: Optional[int] = None, enable_outlier_discounting: bool = False) -> Dict[str, any]:
    """
    Find the optimal turf based on user selection and optional outlier discounting.
    """
    weights = [1 if i == priority_user else 0.5 for i in range(len(user_locations))] if priority_user is not None else None
    return genetic_algorithm_with_outlier_discounting(user_locations, weights=weights, turfs=turfs, enable_outlier_discounting=enable_outlier_discounting)
