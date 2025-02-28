# fwa_optimization.py
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Fitness evaluation function with cross-validation
def evaluate_fitness(C, gamma, X, y):
    model = SVC(C=C, gamma=gamma)
    accuracy = np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1))  # 3-fold cross-validation
    return accuracy

# Fireworks Algorithm for SVM hyperparameter optimization
def fireworks_algorithm(X, y, num_fireworks=10, num_sparks=10, iterations=10, C_range=(0.1, 10), gamma_range=(0.001, 0.1)):
    print("Initializing Fireworks Algorithm...")

    # Initialize random fireworks
    fireworks = [{'C': np.random.uniform(*C_range), 'gamma': np.random.uniform(*gamma_range)} for _ in range(num_fireworks)]

    # Evaluate initial fireworks
    for firework in fireworks:
        firework['fitness'] = evaluate_fitness(firework['C'], firework['gamma'], X, y)

    # Main loop for iterations
    for iteration in range(iterations):
        print(f"Running iteration {iteration + 1}/{iterations}...")
        new_sparks = []
        for firework in fireworks:
            num_local_sparks = int(num_sparks * (firework['fitness'] / sum(f['fitness'] for f in fireworks)))
            for _ in range(num_local_sparks):
                spark_C = np.clip(firework['C'] + np.random.uniform(-1, 1), *C_range)
                spark_gamma = np.clip(firework['gamma'] + np.random.uniform(-0.1, 0.1), *gamma_range)
                new_sparks.append({'C': spark_C, 'gamma': spark_gamma})

        # Evaluate new sparks
        for spark in new_sparks:
            spark['fitness'] = evaluate_fitness(spark['C'], spark['gamma'], X, y)

        # Select the best sparks for the next generation
        fireworks = sorted(fireworks + new_sparks, key=lambda f: f['fitness'], reverse=True)[:num_fireworks]
        print(f"Iteration {iteration+1}/{iterations}, Best Fitness: {fireworks[0]['fitness']:.4f}, C: {fireworks[0]['C']}, Gamma: {fireworks[0]['gamma']}")

    print("Fireworks Algorithm Optimization Completed!")
    return fireworks[0]
