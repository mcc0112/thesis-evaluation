"""
generate_instances.py  –  Circuit-constraint benchmark instances
================================================================

Generates geographic k-nearest Hamiltonian circuit instances following
the method of Francis & Stuckey (2014) "Explaining circuit propagation",
Constraints 19:1-29, Section 3.

Paper model (Fig. 1 of the paper)
-----------------------------------
  - Minimise the length of the longest leg (maxleg).
  - Successor variables must form a Hamiltonian circuit.
  - Only edges present in the transport network may be used
    (travelTime[loc1, loc2] < 0 means no direct connection).
  - The leg-length variable maxleg is bounded by:
      succ[loc1] == loc2  ->  maxleg >= travelTime[loc1, loc2]

Generation procedure (Section 3)
----------------------------------
  1. Place n locations uniformly at random in the unit square.
  2. Compute pairwise Euclidean distances, scaled to integers.
  3. Connect each node to its k nearest geographic neighbours (symmetric).
     The paper uses k = 7; this script keeps k as a CLI parameter so that
     other values can be explored while still following the paper's method.
  4. Perform a random walk to guarantee at least one Hamiltonian circuit:
     whenever the walk gets stuck, add a fresh edge to an unvisited node;
     close the walk back to the start.  Walk-added edges are tracked
     separately so callers can detect near-degenerate instances.
  5. Write a MiniZinc (.mzn) file with:
       - travelTime[Locations, Locations]  (–1 = no direct connection)
       - maxLegLen                         (upper bound on travelTime values)
       - succ[Locations] of var Locations
       - constraint forall(...) using travelTime to block missing edges
       - constraint circuit(succ)
       - var 1..maxLegLen: maxleg
       - constraint forall(...) linking maxleg to travelTime
       - solve minimize maxleg

Walk-edge fraction diagnostic
-------------------------------
The MZN header reports:
  - total undirected edges in the network
  - edges added by the k-nearest step
  - edges added by the random walk (= total − k-nearest)
A high walk fraction indicates that the k-nearest graph was too sparse to
sustain a Hamiltonian circuit on its own, which can make the instance
structurally degenerate.  Callers should flag instances where
walk_edges / total_edges exceeds a chosen threshold (e.g. 0.20).

Usage
------
    python generate_instances.py [options]

Options
-------
  -n, --nodes        INT    Number of locations         (default: 50)
  -k, --neighbours   INT    k-nearest degree            (default: 7)
  -c, --count        INT    Instances to generate       (default: 1)
  -s, --seed         INT    Base random seed            (default: random)
  -o, --outdir       PATH   Output directory            (default: .)
  --scale            INT    Distance scale factor       (default: 1000)
  --prefix           STR    Filename prefix             (default: instance)

Output
------
One .mzn file per instance named  <prefix>_n<N>_k<K>_<index>.mzn
"""

import argparse
import math
import os
import random
import sys
from typing import List, Set, Tuple


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_distance_matrix(
    coords: List[Tuple[float, float]],
    scale: int,
) -> List[List[int]]:
    """Return a symmetric integer distance matrix (scaled Euclidean)."""
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = round(euclidean(coords[i], coords[j]) * scale)
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def k_nearest_edges(
    dist: List[List[int]],
    k: int,
) -> Set[Tuple[int, int]]:
    """
    Return undirected edges {(i,j)} where j is among the k nearest
    neighbours of i.  Stored as (min, max) pairs to avoid duplicates.
    """
    n = len(dist)
    edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        neighbours = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: dist[i][j],
        )
        for j in neighbours[:k]:
            edges.add((min(i, j), max(i, j)))
    return edges


def random_walk_hamiltonian(
    n: int,
    adj: List[Set[int]],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    Perform a random walk to produce a Hamiltonian circuit, adding new
    edges whenever the walk gets stuck.

    Returns the list of (undirected) edges added during the walk so the
    caller can compute the walk-edge fraction.
    """
    start = rng.randrange(n)
    visited = [False] * n
    path = [start]
    visited[start] = True
    added_edges: List[Tuple[int, int]] = []

    current = start
    while len(path) < n:
        unvisited_neighbours = [v for v in adj[current] if not visited[v]]

        if unvisited_neighbours:
            nxt = rng.choice(unvisited_neighbours)
        else:
            unvisited_all = [v for v in range(n) if not visited[v]]
            nxt = rng.choice(unvisited_all)
            adj[current].add(nxt)
            adj[nxt].add(current)
            added_edges.append((min(current, nxt), max(current, nxt)))

        visited[nxt] = True
        path.append(nxt)
        current = nxt

    # Close the circuit back to start
    if start not in adj[current]:
        adj[current].add(start)
        adj[start].add(current)
        added_edges.append((min(current, start), max(current, start)))

    return added_edges


def build_graph(
    n: int,
    k: int,
    dist: List[List[int]],
    rng: random.Random,
) -> Tuple[List[Set[int]], int, int]:
    """
    Build the transport network following Francis & Stuckey Section 3.

    Returns
    -------
    adj              : adjacency sets (adj[i] = set of neighbour indices, 0-based)
    knn_edge_count   : number of undirected edges from the k-nearest step
    walk_edge_count  : number of undirected edges added by the random walk
    """
    knn_edges = k_nearest_edges(dist, k)
    knn_edge_count = len(knn_edges)

    adj: List[Set[int]] = [set() for _ in range(n)]
    for (i, j) in knn_edges:
        adj[i].add(j)
        adj[j].add(i)

    walk_added = random_walk_hamiltonian(n, adj, rng)

    # Count only genuinely new walk edges (not already in k-nearest)
    all_edges = set(knn_edges)
    new_walk_edges = [e for e in walk_added if e not in all_edges]
    walk_edge_count = len(new_walk_edges)

    return adj, knn_edge_count, walk_edge_count


# ---------------------------------------------------------------------------
# MiniZinc writer – minimisation model (paper Fig. 1)
# ---------------------------------------------------------------------------

MZN_TEMPLATE = """\
%% Circuit-constraint benchmark instance
%% Generated by generate_instances.py
%% Method: Francis & Stuckey (2014) "Explaining circuit propagation"
%%
%% Model: minimise the length of the longest leg (maxleg).
%%        Successor variables must form a Hamiltonian circuit using only
%%        edges present in the transport network (travelTime >= 0).
%%        This matches the MiniZinc model in Figure 1 of the paper.
%%
%% Instance parameters
%%   n              = {n}
%%   k              = {k}
%%   seed           = {seed}
%%   knn_edges      = {knn_edges}   (undirected edges from k-nearest step)
%%   walk_edges     = {walk_edges}  (undirected edges added by random walk)
%%   total_edges    = {total_edges}
%%   walk_fraction  = {walk_fraction:.4f}  (walk_edges / total_edges)
%%
%% NOTE: a walk_fraction above 0.20 may indicate a near-degenerate instance
%%       in which the random walk dominates the graph structure.

include "globals.mzn";

int: n = {n};
set of int: Locations = 1..n;

%% Maximum possible travel time; used as the upper bound for maxleg.
int: maxLegLen = {max_leg_len};

%% Travel times between locations (–1 means no direct connection exists).
array[Locations, Locations] of int: travelTime =
  array2d(Locations, Locations, [ {travel_time_rows} ]);


%% Successor variables: succ[i] = next location after i in the tour.
array[Locations] of var Locations: succ;

%% Only use allowed legs (edges present in the transport network).
constraint forall(loc1, loc2 in Locations)(
  travelTime[loc1, loc2] < 0 -> succ[loc1] != loc2
);

%% Successors must form a Hamiltonian circuit.
constraint circuit(succ);

%% Variable for the length of the longest leg.
var 1..maxLegLen: maxleg;

%% Link maxleg to actual travel times used in the tour.
constraint forall(loc1, loc2 in Locations)(
  succ[loc1] = loc2 -> maxleg >= travelTime[loc1, loc2]
);

solve :: int_search(succ, input_order, indomain_min, complete)
      minimize maxleg;

output [
  "succ = ", show(succ), "\\n",
  "maxleg = ", show(maxleg), "\\n"
];
"""


def format_travel_time(dist, adj, n):
    """
    Produce a flat list for array2d and return (flat_list_str, max_tt).
    """
    flat = []
    max_tt = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                flat.append(-1)
            elif j in adj[i]:
                tt = dist[i][j]
                flat.append(tt)
                if tt > max_tt:
                    max_tt = tt
            else:
                flat.append(-1)

    # Convert to comma-separated string
    flat_str = ", ".join(str(x) for x in flat)
    return flat_str, max_tt



def write_mzn(
    filepath: str,
    n: int,
    k: int,
    seed: int,
    dist: List[List[int]],
    adj: List[Set[int]],
    knn_edges: int,
    walk_edges: int,
) -> None:
    total_edges = knn_edges + walk_edges
    walk_fraction = walk_edges / total_edges if total_edges > 0 else 0.0

    travel_time_rows, max_leg_len = format_travel_time(dist, adj, n)

    content = MZN_TEMPLATE.format(
        n=n,
        k=k,
        seed=seed,
        knn_edges=knn_edges,
        walk_edges=walk_edges,
        total_edges=total_edges,
        walk_fraction=walk_fraction,
        max_leg_len=max_leg_len,
        travel_time_rows=travel_time_rows,
    )

    with open(filepath, "w") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_instance(
    n: int,
    k: int,
    seed: int,
    scale: int,
) -> Tuple[List[List[int]], List[Set[int]], int, int]:
    """
    Generate one instance.

    Returns
    -------
    dist             : integer distance matrix (0-based indices)
    adj              : adjacency sets (0-based indices)
    knn_edge_count   : edges from k-nearest step
    walk_edge_count  : edges added by random walk
    """
    rng = random.Random(seed)
    coords = [(rng.random(), rng.random()) for _ in range(n)]
    dist = build_distance_matrix(coords, scale)
    adj, knn_edge_count, walk_edge_count = build_graph(n, k, dist, rng)
    return dist, adj, knn_edge_count, walk_edge_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate circuit-constraint benchmark instances "
            "(Francis & Stuckey 2014, Fig. 1) and write MiniZinc files. "
            "Model: minimise the length of the longest tour leg."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--nodes",      type=int, default=50,
                        help="Number of locations")
    parser.add_argument("-k", "--neighbours", type=int, default=7,
                        help="k-nearest neighbour degree (paper uses 7)")
    parser.add_argument("-c", "--count",      type=int, default=1,
                        help="Number of instances to generate")
    parser.add_argument("-s", "--seed",       type=int, default=None,
                        help="Base random seed (random if omitted)")
    parser.add_argument("-o", "--outdir",     type=str, default=".",
                        help="Output directory")
    parser.add_argument("--scale",            type=int, default=1000,
                        help="Distance scale factor applied to Euclidean distances")
    parser.add_argument("--prefix",           type=str, default="instance",
                        help="Filename prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randrange(2 ** 32)

    print(f"Generating {args.count} instance(s) (minimise maxleg, Francis & Stuckey 2014):")
    print(f"  n={args.nodes}, k={args.neighbours}, scale={args.scale}")
    print(f"  base seed={base_seed}, output dir='{args.outdir}'")
    print()

    for idx in range(args.count):
        seed = base_seed + idx

        dist, adj, knn_edges, walk_edges = generate_instance(
            n=args.nodes,
            k=args.neighbours,
            seed=seed,
            scale=args.scale,
        )

        total = knn_edges + walk_edges
        walk_frac = walk_edges / total if total > 0 else 0.0

        filename = f"{args.prefix}_n{args.nodes}_k{args.neighbours}_{idx:04d}.mzn"
        filepath = os.path.join(args.outdir, filename)

        write_mzn(
            filepath=filepath,
            n=args.nodes,
            k=args.neighbours,
            seed=seed,
            dist=dist,
            adj=adj,
            knn_edges=knn_edges,
            walk_edges=walk_edges,
        )

        flag = "  *** HIGH WALK FRACTION ***" if walk_frac > 0.20 else ""
        print(
            f"  [{idx:4d}] seed={seed:10d}  "
            f"knn_edges={knn_edges:5d}  walk_edges={walk_edges:3d}  "
            f"walk_frac={walk_frac:.3f}{flag}  -> {filepath}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()