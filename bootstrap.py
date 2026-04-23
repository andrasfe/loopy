"""Monte-Carlo loop: sample K puzzles per generation, keep top parents, iterate."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import config
import evaluator
import proposer
import solver


def _format_grid(grid: list[list[int]]) -> str:
    lines = []
    for r, row in enumerate(grid):
        cells = []
        for c, v in enumerate(row):
            cells.append(str(v) if v else ".")
            if c in (2, 5):
                cells.append("|")
        lines.append(" ".join(cells))
        if r in (2, 5):
            lines.append("-" * 21)
    return "\n".join(lines)


def _score_summary(s: evaluator.Score) -> str:
    if not s.valid:
        return f"INVALID      fit={s.fitness:>5} clues={s.num_clues}"
    if not s.unique:
        return f"NON-UNIQUE   fit={s.fitness:>5} clues={s.num_clues}"
    return (
        f"diff={s.difficulty}  dist={int(s.target_distance)}  "
        f"clues={s.num_clues}  bt={s.backtracks}  fit={int(s.fitness)}"
    )


def run(
    target: int,
    k: int,
    max_generations: int,
    keep: int,
    seed: Optional[int],
    log_path: Path,
) -> Optional[evaluator.Score]:
    if seed is not None:
        random.seed(seed)
    settings = config.load()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")

    # Fixed seed grid for this whole run: every proposal masks this same solved Sudoku.
    seed_grid = solver.random_solved(seed=seed)

    def record(event: dict):
        log.write(json.dumps(event, default=str) + "\n")
        log.flush()

    record({
        "event": "start",
        "model": settings.model,
        "target": target, "k": k, "max_generations": max_generations, "keep": keep,
        "seed_grid": seed_grid,
        "time": time.time(),
    })
    print(f"[loopy] model={settings.model}  target={target}/10  K={k}  keep={keep}")

    # Generation 0: cold sampling
    population: list[evaluator.Score] = []
    total_tokens_in = 0
    total_tokens_out = 0
    best_ever: Optional[evaluator.Score] = None
    stagnant_gens = 0
    prev_best_fitness: float = float("-inf")
    STAGNATION_PATIENCE = 2  # if best doesn't improve for this many gens, reset parents

    for gen in range(max_generations):
        t0 = time.time()
        if gen == 0 or not population or population[0].grid is None:
            proposals = proposer.fresh(seed_grid=seed_grid, k=k, target=target)
            mode = "fresh"
        elif stagnant_gens >= STAGNATION_PATIENCE:
            # Escape local optimum: abandon parents, generate fresh, keep best-ever as memory.
            proposals = proposer.fresh(seed_grid=seed_grid, k=k, target=target)
            mode = "reboot"
            population = []
            stagnant_gens = 0
            prev_best_fitness = float("-inf")  # let the new lineage be judged fresh
        else:
            per_parent = max(1, k // len(population))
            proposals = []
            for parent in population:
                proposals.extend(
                    proposer.mutate(
                        seed_grid=seed_grid,
                        parent_grid=parent.grid,
                        parent_feedback=parent.feedback,
                        parent_clues=parent.num_clues,
                        parent_difficulty=parent.difficulty,
                        k=per_parent,
                        target=target,
                    )
                )
            mode = "mutate"

        scored: list[evaluator.Score] = []
        for p in proposals:
            total_tokens_in += p.tokens_in
            total_tokens_out += p.tokens_out
            if p.error:
                scored.append(evaluator.Score(
                    grid=None, valid=False, unique=False, solvable=False,
                    difficulty=0, num_clues=0,
                    feedback=f"API error: {p.error}", raw="",
                ))
                continue
            scored.append(evaluator.evaluate(p.text, target_difficulty=target))

        # Merge into population and keep top-`keep`.
        combined = population + scored
        combined.sort(key=lambda s: s.fitness, reverse=True)
        population = combined[:keep]

        # Track best + stagnation
        gen_best = max(scored, key=lambda s: s.fitness) if scored else None
        if gen_best and (best_ever is None or gen_best.fitness > best_ever.fitness):
            best_ever = gen_best
        if population[0].fitness > prev_best_fitness + 1e-6:
            stagnant_gens = 0
            prev_best_fitness = population[0].fitness
        else:
            stagnant_gens += 1

        dt = time.time() - t0
        valid_count = sum(1 for s in scored if s.valid)
        unique_count = sum(1 for s in scored if s.unique)
        on_target = sum(1 for s in scored if s.unique and s.difficulty == target)
        print(
            f"[gen {gen:>2} {mode:<6}]  "
            f"valid={valid_count}/{len(scored)}  unique={unique_count}  "
            f"on-target={on_target}  "
            f"best={_score_summary(population[0])}  "
            f"dt={dt:.1f}s"
        )
        record({
            "event": "generation",
            "gen": gen, "mode": mode,
            "valid": valid_count, "unique": unique_count, "on_target": on_target,
            "best_fitness": population[0].fitness,
            "best_difficulty": population[0].difficulty,
            "best_grid": population[0].grid,
            "dt": dt,
            "tokens_in": total_tokens_in, "tokens_out": total_tokens_out,
        })

        if population[0].unique and population[0].difficulty == target:
            print(f"\n[loopy] Converged at generation {gen}. Target difficulty {target} hit.")
            record({"event": "converged", "gen": gen, "grid": population[0].grid})
            log.close()
            return population[0]

    print(f"\n[loopy] Max generations reached without exact convergence.")
    if best_ever:
        print(f"Best ever: {_score_summary(best_ever)}")
    record({
        "event": "max_generations",
        "best_ever_fitness": best_ever.fitness if best_ever else None,
        "best_ever_difficulty": best_ever.difficulty if best_ever else None,
        "best_ever_grid": best_ever.grid if best_ever else None,
    })
    log.close()
    return best_ever


def main():
    ap = argparse.ArgumentParser(description="MC-sampled Sudoku generator.")
    ap.add_argument("--target", type=int, default=7, help="target difficulty 1-10")
    ap.add_argument("-k", type=int, default=6, help="proposals per generation")
    ap.add_argument("--keep", type=int, default=2, help="parents kept between generations")
    ap.add_argument("--max-gen", type=int, default=8, help="max generations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", type=str, default=None, help="output jsonl log path")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else Path("runs") / f"run-{int(time.time())}.jsonl"

    result = run(
        target=args.target,
        k=args.k,
        max_generations=args.max_gen,
        keep=args.keep,
        seed=args.seed,
        log_path=log_path,
    )

    print("\n" + "=" * 40)
    if result and result.unique and result.grid is not None:
        print(f"Difficulty: {result.difficulty}/10   Clues: {result.num_clues}")
        print(f"Techniques: {result.techniques}  Backtracks: {result.backtracks}")
        print()
        print(_format_grid(result.grid))
    else:
        print("No valid unique puzzle found.")
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
