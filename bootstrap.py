"""Monte-Carlo loop: LLM writes code, evaluator scores the output, iterate."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

import proposer
from score import Score


def _load_evaluator(task_path: Path):
    """Load the evaluator module specified in the task's YAML frontmatter.
    Falls back to 'sudoku' if not specified."""
    text = task_path.read_text()
    evaluator_name = "sudoku"
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            front = text[3:end]
            m = re.search(r"^evaluator:\s*(\S+)", front, re.MULTILINE)
            if m:
                evaluator_name = m.group(1)
    return importlib.import_module(f"evaluators.{evaluator_name}")


def _score_summary(s: Score) -> str:
    if s.exec_error:
        return f"EXEC-ERR     fit={s.fitness:>5}"
    if not s.valid:
        return f"INVALID      fit={s.fitness:>5}"
    return (
        f"level={s.difficulty}  dist={int(s.target_distance)}  "
        f"fit={int(s.fitness)}"
    )


def run(
    task_path: Path,
    target: int,
    k: int,
    max_generations: int,
    keep: int,
    seed: Optional[int],
    log_path: Path,
) -> Optional[Score]:
    proposer.init(task_path)
    evaluator = _load_evaluator(task_path)
    if seed is not None:
        random.seed(seed)
    settings = __import__("config").load()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")

    def record(event: dict):
        log.write(json.dumps(event, default=str) + "\n")
        log.flush()

    record({
        "event": "start",
        "task": str(task_path),
        "model": settings.model,
        "target": target, "k": k, "max_generations": max_generations, "keep": keep,
        "time": time.time(),
    })
    print(f"[loopy] task={task_path.name}  model={settings.model}  target={target}/10  K={k}  keep={keep}")

    population: list[Score] = []
    total_tokens_in = 0
    total_tokens_out = 0
    best_ever: Optional[Score] = None
    stagnant_gens = 0
    prev_best_fitness: float = float("-inf")
    STAGNATION_PATIENCE = 2

    for gen in range(max_generations):
        t0 = time.time()
        if gen == 0 or not population or population[0].code is None:
            proposals = proposer.fresh(k=k, target=target)
            mode = "fresh"
        elif stagnant_gens >= STAGNATION_PATIENCE:
            proposals = proposer.fresh(k=k, target=target)
            mode = "reboot"
            population = []
            stagnant_gens = 0
            prev_best_fitness = float("-inf")
        else:
            per_parent = max(1, k // len(population))
            proposals = []
            for parent in population:
                proposals.extend(
                    proposer.mutate(
                        parent_code=parent.code,
                        parent_feedback=parent.feedback,
                        parent_exec_error=parent.exec_error,
                        parent_difficulty=parent.difficulty,
                        k=per_parent,
                        target=target,
                    )
                )
            mode = "mutate"

        scored: list[Score] = []
        for p in proposals:
            total_tokens_in += p.tokens_in
            total_tokens_out += p.tokens_out
            if p.error:
                scored.append(Score(
                    code=None, valid=False, difficulty=0,
                    feedback=f"API error: {p.error}",
                    exec_error=f"API error: {p.error}",
                ))
                continue
            if p.code is None:
                scored.append(Score(
                    code=None, valid=False, difficulty=0,
                    feedback="Could not extract Python code from LLM response.",
                    raw=p.text,
                    exec_error="No code extracted",
                ))
                continue
            scored.append(evaluator.evaluate_code(p.code, target=target))

        combined = population + scored
        combined.sort(key=lambda s: s.fitness, reverse=True)
        population = combined[:keep]

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
        on_target = sum(1 for s in scored if s.valid and s.target_distance == 0)
        exec_ok = sum(1 for s in scored if not s.exec_error)
        gen_tokens_in = sum(p.tokens_in for p in proposals)
        gen_tokens_out = sum(p.tokens_out for p in proposals)
        print(
            f"[gen {gen:>2} {mode:<6}]  "
            f"exec={exec_ok}/{len(scored)}  valid={valid_count}  "
            f"on-target={on_target}  "
            f"best={_score_summary(population[0])}  "
            f"tok={gen_tokens_in}+{gen_tokens_out}  "
            f"dt={dt:.1f}s"
        )
        record({
            "event": "generation",
            "gen": gen, "mode": mode,
            "exec_ok": exec_ok, "valid": valid_count,
            "on_target": on_target,
            "best_fitness": population[0].fitness,
            "best_difficulty": population[0].difficulty,
            "best_details": population[0].details,
            "dt": dt,
            "tokens_in": total_tokens_in, "tokens_out": total_tokens_out,
        })

        if population[0].valid and population[0].target_distance == 0:
            print(
                f"\n[loopy] Converged at generation {gen}. Target {target} hit."
                f"\n[loopy] Total tokens: {total_tokens_in} in + {total_tokens_out} out = {total_tokens_in + total_tokens_out}"
            )
            record({"event": "converged", "gen": gen, "details": population[0].details,
                     "total_tokens_in": total_tokens_in, "total_tokens_out": total_tokens_out})
            log.close()
            return population[0]

    print(f"\n[loopy] Max generations reached without exact convergence.")
    print(f"[loopy] Total tokens: {total_tokens_in} in + {total_tokens_out} out = {total_tokens_in + total_tokens_out}")
    if best_ever:
        print(f"Best ever: {_score_summary(best_ever)}")
    record({
        "event": "max_generations",
        "best_ever_fitness": best_ever.fitness if best_ever else None,
        "best_ever_difficulty": best_ever.difficulty if best_ever else None,
        "best_ever_details": best_ever.details if best_ever else None,
        "total_tokens_in": total_tokens_in, "total_tokens_out": total_tokens_out,
    })
    log.close()
    return best_ever


def main():
    ap = argparse.ArgumentParser(description="Self-improving LLM code generation loop.")
    ap.add_argument("--task", type=str, default="tasks/sudoku.md", help="path to task prompt .md file")
    ap.add_argument("--target", type=int, default=7, help="target level 1-10")
    ap.add_argument("-k", type=int, default=6, help="proposals per generation")
    ap.add_argument("--keep", type=int, default=2, help="parents kept between generations")
    ap.add_argument("--max-gen", type=int, default=8, help="max generations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", type=str, default=None, help="output jsonl log path")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else Path("runs") / f"run-{int(time.time())}.jsonl"

    result = run(
        task_path=Path(args.task),
        target=args.target,
        k=args.k,
        max_generations=args.max_gen,
        keep=args.keep,
        seed=args.seed,
        log_path=log_path,
    )

    print("\n" + "=" * 40)
    if result and result.valid and result.target_distance == 0:
        print(f"Level: {result.difficulty}/10")
        if result.details:
            for k, v in result.details.items():
                print(f"  {k}: {v}")
        if result.code:
            print("\n--- Winning code ---")
            print(result.code)
    else:
        print("Did not converge.")
        if result:
            print(f"Best: level={result.difficulty}, fitness={result.fitness:.1f}")
            if result.details:
                for k, v in result.details.items():
                    print(f"  {k}: {v}")
            if result.code:
                print("\n--- Best code ---")
                print(result.code)
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
