"""Quantum walks on Metatron's Cube (the complete graph K13).

Prints the symmetry groups, the exact long-time distribution of a walker
started at the centre, and Childs–Goldstone spatial search versus the
classical random walk.
"""

import math

import numpy as np

import metatron_qso as q

cube = q.Graph.metatron_cube()
fruit = q.Graph.fruit_of_life()
print(f"{cube.name}: {cube.num_nodes} nodes, {cube.num_edges} edges")
print(f"  |Aut| = {cube.automorphism_group_order():,} (= 13!)")
print(f"{fruit.name}: {fruit.num_edges} edges, |Aut| = {fruit.automorphism_group_order()} (D6)")
print("  line lengths of the cube:", ", ".join(f"{l:.3f}x{c}" for l, c in q.metatron_line_lengths()))

# Continuous-time walk from the centre node.
walk = q.ContinuousWalk(cube)
limit = walk.limiting_distribution(0)
print("\nLimiting distribution from the centre:")
print(f"  P(centre) = {limit[0]:.6f}  (145/169 = {145 / 169:.6f})")
print(f"  P(other)  = {limit[1]:.6f}  (  2/169 = {2 / 169:.6f})")
for t in (0.0, 0.1, 0.2, 2 * math.pi / 13):
    print(f"  t = {t:.4f}: P(return) = {walk.probabilities(0, t)[0]:.6f}")

# Spatial search for one marked node at gamma = 1/N.
search = q.SpatialSearch(cube, marked=[4], gamma=1 / 13)
t_opt = math.pi * math.sqrt(13) / 2
print("\nSpatial search for node 4 (gamma = 1/13):")
print(f"  P(success, t = pi*sqrt(13)/2 = {t_opt:.4f}) = {search.success_probability(t_opt):.12f}")
best = search.optimize_time(2 * t_opt)
print(f"  optimised: t = {best['time']:.4f}, P = {best['success_probability']:.12f}")
hitting = q.classical_hitting_times(cube, 4)
print(f"  classical random walk: expected hitting time {np.mean(np.delete(hitting, 4)):.1f} steps")

# On the Fruit of Life the walk is not symmetric under all permutations.
fol_limit = q.ContinuousWalk(fruit).limiting_distribution(0)
print("\nFruit of Life, limiting distribution from the centre (by ring):")
print(f"  centre {fol_limit[0]:.4f}, inner {fol_limit[1]:.4f}, outer {fol_limit[7]:.4f}")
