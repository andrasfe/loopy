---
name: Max-Cut
description: Find a maximum weight cut in a fixed weighted graph
evaluator: maxcut
---

You are a combinatorial optimization ENGINEER. You write fully self-contained Python scripts that solve the Maximum Cut problem.

PROBLEM:
  Given an undirected weighted graph, partition the nodes into two sets S and T
  to maximize the total weight of edges crossing the partition (one endpoint in S,
  the other in T).

GRAPH (20 nodes, 50 edges):
  Edges as (u, v, weight):
  (0,1,7) (0,4,3) (0,7,5) (0,12,8) (1,2,6) (1,5,4) (1,9,9) (2,3,5) (2,6,7) (2,10,3)
  (3,7,8) (3,11,6) (3,14,4) (4,5,9) (4,8,5) (4,13,7) (5,6,3) (5,9,8) (6,7,6) (6,10,5)
  (7,11,4) (7,15,9) (8,9,7) (8,12,3) (8,16,6) (9,10,5) (9,13,8) (10,11,9) (10,14,4) (10,17,7)
  (11,15,3) (11,18,5) (12,13,6) (12,16,8) (13,14,9) (13,17,4) (14,15,7) (14,18,3) (15,19,6)
  (16,17,5) (16,19,9) (17,18,8) (17,19,4) (18,19,7) (0,19,5) (1,18,6) (2,17,4) (3,16,8) (5,15,3) (6,14,9)

RULES:
  - Use ONLY the Python standard library (json, random, itertools, etc.)
  - Your script must implement everything from scratch.
  - Must print a JSON object to stdout with:
      {"partition": [0,1,0,1,...], "cut_value": N}
    where partition is a list of 20 integers (0 or 1), and cut_value is the
    total weight of edges crossing the partition.
  - Both sets must be non-empty (not all 0s or all 1s).
  - The script must complete within 10 seconds.

STRATEGY HINTS:
  - Random partition gives ~50% of optimal. Not enough for high targets.
  - Greedy: start with random partition, flip each node, keep if cut improves.
  - Local search with restarts: repeat greedy from many random starts.
  - Simulated annealing: accept worse moves with decreasing probability.
  - The total edge weight is 299. Optimal cut is 247.

OUTPUT: Respond with ONLY a Python code block. No prose, no explanation.
