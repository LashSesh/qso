"""QAOA for MaxCut with exact gradients.

Reproduces two closed-form results for depth p = 1 — the 3/4 ratio on rings
(Farhi et al. 2014) and 0.6924 on 3-regular triangle-free graphs such as
the cube Q3 — then optimises layer by layer with INTERP initialisation.
"""

import metatron_qso as q

for graph, reference in [(q.Graph.cycle(8), 0.75), (q.Graph.hypercube(3), 0.6924500897)]:
    result = q.MaxCutQaoa(graph, 1).optimize(restarts=8, seed=0)
    print(
        f"{graph.name:4s} p=1: <C> = {result['expectation']:.6f} of {result['max_cut']:.0f}, "
        f"ratio = {result['approximation_ratio']:.8f} (theory {reference})"
    )

graph = q.Graph.petersen()
print(f"\n{graph.name}: layerwise optimisation (max cut {q.MaxCutQaoa(graph, 1).max_cut:.0f})")
for r in q.optimize_qaoa_layerwise(graph, p_max=4, restarts=4, seed=1):
    print(
        f"  p={r['p']}: ratio {r['approximation_ratio']:.6f}, "
        f"P(optimal cut) = {r['optimal_probability']:.4f}, "
        f"most likely {r['most_likely_bitstring']} (cut {r['most_likely_cut']:.0f})"
    )

# The same QAOA state as an explicit gate circuit, e.g. for export.
circuit = q.qaoa_maxcut_circuit(q.Graph.cycle(8), 1)
best = q.MaxCutQaoa(q.Graph.cycle(8), 1).optimize()
params = list(best["gammas"]) + list(best["betas"])
energy = q.simulate(circuit, params).expectation(q.maxcut_cost(q.Graph.cycle(8)))
print(f"\nGate-level circuit: {len(circuit)} gates, depth {circuit.depth}, <C> = {energy:.6f}")
