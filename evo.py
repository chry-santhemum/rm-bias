import json
import dotenv
import random
from typing import Literal
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from sklearn.metrics import pairwise_distances

from caller import ChatHistory
from state import SeedState, AttributeStats, Rollout
from utils import parse_json_response, set_seed_all
from cluster_models import ClusterModel
from api_models import GenerationModel
from reward_models import RewardModel
from runner import Runner
from bias_evaluator import BiasEvaluator
from planner import Planner

from evo_prompts import *

dotenv.load_dotenv()
set_seed_all(10086)


class EvoPlanner:
    def __init__(
        self,
        direction: Literal["plus", "minus"],
        hypothesis_planner: Planner, 
        cluster_model: ClusterModel,
    ):
        self.direction: Literal["plus", "minus"] = direction
        self.hypothesis_planner = hypothesis_planner
        self.cluster_model = cluster_model

    async def initial_plan(
        self,
        runner: Runner,
    ):
        return await self.hypothesis_planner.plan(
            runner=runner, 
            direction=self.direction, 
            cluster_model=self.cluster_model
        )

    def _collect_all_evaluated_attributes(
        self,
        seed_state: SeedState,
    ) -> list[dict]:
        """
        Collect all previously evaluated attributes from history with their winrates.
        Returns list of dicts with keys: attribute, student_winrate, teacher_winrate
        """
        evaluated = []
        for time_step, history_entry in enumerate(seed_state.history):
            for attribute, stats in history_entry.items():
                student_wr = stats.winrate("student")
                teacher_wr = stats.winrate("teacher")
                # Only include if we have valid winrates
                if student_wr is not None and teacher_wr is not None:
                    evaluated.append({
                        "attribute": attribute,
                        "student_winrate": student_wr,
                        "teacher_winrate": teacher_wr,
                        "time_step": time_step,
                    })
        return evaluated

    def _get_nearest_neighbors(
        self,
        target_attribute: str,
        all_evaluated: list[dict],
        n_neighbors: int = 16,
    ) -> str:
        """
        Find the n_neighbors closest attributes to target_attribute in embedding space.
        Returns formatted string for the prompt.
        """
        if not all_evaluated:
            return "No similar attributes available."

        # Get all attribute strings including target
        all_attributes = [target_attribute] + [e["attribute"] for e in all_evaluated]

        # Embed all attributes
        embeddings = self.cluster_model.embed(all_attributes)

        # Compute distances from target (index 0) to all others
        target_embedding = embeddings[0:1]
        other_embeddings = embeddings[1:]
        distances = pairwise_distances(target_embedding, other_embeddings, metric="cosine")[0]

        # Get indices of nearest neighbors (excluding exact matches with distance ~0)
        sorted_indices = np.argsort(distances)

        neighbors = []
        for idx in sorted_indices:
            if len(neighbors) >= n_neighbors:
                break
            # Skip if it's the exact same attribute
            if all_evaluated[idx]["attribute"] == target_attribute:
                continue
            neighbors.append(all_evaluated[idx])

        # Format as string
        lines = []
        for i, neighbor in enumerate(neighbors, 1):
            lines.append(
                f"{i}. Attribute: {neighbor['attribute']}\n"
                f"   Metric A winrate: {neighbor['student_winrate']:.3f}\n"
                f"   Metric B winrate: {neighbor['teacher_winrate']:.3f}"
            )

        return "\n".join(lines) if lines else "No similar attributes available."

    async def iterate_plan(
        self,
        seed_states: list[SeedState],
        m_var: int,
        n_neighbors: int = 16,
    ):
        to_send_messages = []
        messages_info = []

        for seed_idx, seed_state in enumerate(seed_states):
            # Collect all previously evaluated attributes for neighbor lookup
            all_evaluated = self._collect_all_evaluated_attributes(seed_state)

            for attribute, time_step in seed_state.state.items():
                # Get winrates for the current attribute
                stats = seed_state.history[time_step][attribute]
                student_wr = stats.winrate("student")
                teacher_wr = stats.winrate("teacher")

                # Format winrates (handle None case)
                student_wr_str = f"{student_wr:.3f}" if student_wr is not None else "N/A"
                teacher_wr_str = f"{teacher_wr:.3f}" if teacher_wr is not None else "N/A"

                planner_prompt = MUTATE_PROMPT.format(
                    num_plans=m_var,
                    cluster_summary=seed_state.cluster.summary,
                    original_attribute=attribute,
                    student_winrate=student_wr_str,
                    teacher_winrate=teacher_wr_str,
                    neighbor_data=self._get_nearest_neighbors(
                        target_attribute=attribute,
                        all_evaluated=all_evaluated,
                        n_neighbors=n_neighbors,
                    ),
                    direction_goal=DIRECTION_GOAL[self.direction],
                    bias_nudge=BIAS_NUDGE[self.direction],
                )

                to_send_messages.append(
                    ChatHistory.from_user(planner_prompt)
                )
                messages_info.append(
                    {
                        "parent": attribute,
                        "parent_time_step": time_step,
                        "seed_idx": seed_idx,
                    }
                )

            seed_state.history.append({})

        planner_responses = await self.hypothesis_planner.sample(to_send_messages, desc="Mutating", enable_cache=False)

        # parse responses
        for i, resp in enumerate(planner_responses):
            if resp is None:
                continue
            seed_idx = messages_info[i]["seed_idx"]
            attributes, reasoning = parse_json_response(resp)
            if isinstance(attributes, str):
                attributes = []
            elif isinstance(attributes, list):
                attributes = [p.strip() for p in attributes]

            if i < 3:
                logger.info(f"Planner reasoning:\n{reasoning}")
                logger.info(f"Planner prompt:\n{to_send_messages[i].get_first('user')}")
                logger.info(f"Planner attributes:\n{json.dumps(attributes, indent=4)}")

            meta = {
                "time_step": len(seed_states[seed_idx].history) - 1,
                "parent": messages_info[i]["parent"],
                "parent_time_step": messages_info[i]["parent_time_step"],
                "operation": "mutate",
                "planner_model": self.hypothesis_planner.curr_planner_model,
                "reasoning_effort": str(self.hypothesis_planner.reasoning) if self.hypothesis_planner.reasoning else None,
                "planner_prompt": to_send_messages[i].get_first("user"),
                "planner_reasoning": str(reasoning),
                "m_var": m_var,
            }

            for attribute in attributes:
                seed_states[seed_idx].history[-1][attribute] = AttributeStats(
                    attribute=attribute,
                    rollouts={},
                    meta=meta,
                )

        for seed_state in seed_states:
            # add in the original prompts from the state
            # reason: need to re-evaluate on different user prompts
            for attribute, time_step in seed_state.state.items():
                seed_state.history[-1][attribute] = AttributeStats(
                    attribute=attribute,
                    rollouts={},
                    meta=seed_state.history[time_step][attribute].meta,
                )

            logger.info(f"Seed {seed_state.index} mutated plus original plans: {len(seed_state.history[-1])}")
    
    @staticmethod
    def _dominates(p1: tuple, p2: tuple) -> bool:
        """Returns True if p1 dominates p2."""
        r1_a, r2_a = p1[2], p1[3]
        r1_b, r2_b = p2[2], p2[3]
        return (r1_a >= r1_b and r2_a <= r2_b) and (r1_a > r1_b or r2_a < r2_b)

    @staticmethod
    def _get_pareto_fronts(candidates: list[tuple]) -> list[list[tuple]]:
        """Sorts candidates into Pareto fronts."""
        n = len(candidates)
        domination_counts = [0] * n  # number of candidates that dominates each candidate
        dominated_sets = [[] for _ in range(n)]  # candiates that each candidate dominates
        
        # Calculate dominance
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if EvoPlanner._dominates(candidates[i], candidates[j]):
                    dominated_sets[i].append(j)
                elif EvoPlanner._dominates(candidates[j], candidates[i]):
                    domination_counts[i] += 1

        fronts = []
        current_front = [i for i, count in enumerate(domination_counts) if count == 0]
        
        rank = 0
        while current_front:
            fronts.append([candidates[i] for i in current_front])
            next_front = []
            for i in current_front:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            rank += 1
            current_front = next_front
            
        return fronts

    @staticmethod
    def _pareto_tiebreak(candidate: tuple) -> float:
        """smaller is better"""
        r1, r2 = candidate[2], candidate[3]
        return -r1
        # return np.sqrt((1.0 - r1)**2 + (r2 - 0.0)**2)

    @staticmethod
    def plot_candidate_stats(
        all_candidates: list[tuple],
        filtered_candidates: list[tuple],
        selected_candidates: list[tuple],
        student_threshold: float,
        teacher_threshold: float,
        direction: Literal["plus", "minus"],
    ) -> matplotlib.figure.Figure:
        """Creates a scatterplot of candidate statistics."""
        filtered_set = set(id(c) for c in filtered_candidates)
        selected_set = set(id(c) for c in selected_candidates)

        # Three categories: filtered out, passed but not selected, selected
        filtered_out = [(c[2], c[3]) for c in all_candidates
                        if c[2] is not None and c[3] is not None and id(c) not in filtered_set]
        passed_not_selected = [(c[2], c[3]) for c in all_candidates
                               if c[2] is not None and c[3] is not None
                               and id(c) in filtered_set and id(c) not in selected_set]
        selected = [(c[2], c[3]) for c in all_candidates
                    if c[2] is not None and c[3] is not None and id(c) in selected_set]

        fig, ax = plt.subplots(figsize=(10, 8))
        if filtered_out:
            ax.scatter([p[0] for p in filtered_out], [p[1] for p in filtered_out],
                       c='gray', alpha=0.4, label='Filtered out', marker='x', s=30)
        if passed_not_selected:
            ax.scatter([p[0] for p in passed_not_selected], [p[1] for p in passed_not_selected],
                       c='blue', alpha=0.5, label='Passed filter', marker='o', s=40)
        if selected:
            ax.scatter([p[0] for p in selected], [p[1] for p in selected],
                       c='red', alpha=0.9, label='Selected', marker='o', s=60)

        # Vertical line for student threshold
        ax.axvline(x=student_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Student threshold ({student_threshold:.3f})')
        # Horizontal line for teacher threshold
        ax.axhline(y=teacher_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Teacher threshold ({teacher_threshold:.3f})')

        # Shade the rejection regions based on direction
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if direction == "plus":
            # Reject low student scores (left of threshold)
            ax.axvspan(xlim[0], student_threshold, alpha=0.1, color='red')
            # Reject high teacher scores (above threshold)
            ax.axhspan(teacher_threshold, ylim[1], alpha=0.1, color='red')
        else:
            # Reject high student scores (right of threshold)
            ax.axvspan(student_threshold, xlim[1], alpha=0.1, color='red')
            # Reject low teacher scores (below threshold)
            ax.axhspan(ylim[0], teacher_threshold, alpha=0.1, color='red')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('Student Winrate/RewardDiff')
        ax.set_ylabel('Teacher Winrate/RewardDiff')
        ax.set_title('Candidate Stats: Student vs Teacher')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        return fig

    def update_pop_pareto(self, seed_states: list[SeedState], n_pop_target: int) -> dict[int, dict]:
        if self.direction == "minus":
            raise NotImplementedError
        logger.info(f"Trying to update population to {n_pop_target} members using Pareto front selection.")

        candidates_by_seed: dict[int, dict] = {}

        for seed_state in seed_states:
            all_candidates = []  # All candidates before filtering
            for attribute, stats in seed_state.history[-1].items():
                student_winrate = stats.winrate("student")
                teacher_winrate = stats.winrate("teacher")
                new_candidate = (
                    attribute,
                    stats.meta["time_step"],
                    student_winrate,
                    teacher_winrate,
                )
                # print(f"Attribute:\n{attribute}\nStudent winrate: {new_candidate[2]}\nTeacher winrate: {new_candidate[3]}\n")

                all_candidates.append(new_candidate)

            # Calculate percentile-based thresholds
            valid_student_winrates = [c[2] for c in all_candidates if c[2] is not None]
            valid_teacher_winrates = [c[3] for c in all_candidates if c[3] is not None]

            if self.direction == "plus":
                # Filter out bottom 25% of student scores, but don't filter positive scores
                student_pct = float(np.percentile(valid_student_winrates, 25)) if valid_student_winrates else float('-inf')
                student_threshold = min(0.0, student_pct)
                # Filter out top 25% of teacher scores, but don't filter negative scores
                teacher_pct = float(np.percentile(valid_teacher_winrates, 75)) if valid_teacher_winrates else float('inf')
                teacher_threshold = max(0.0, teacher_pct)
            else:
                # For minus direction: opposite logic
                student_pct = float(np.percentile(valid_student_winrates, 75)) if valid_student_winrates else float('inf')
                student_threshold = max(0.0, student_pct)
                teacher_pct = float(np.percentile(valid_teacher_winrates, 25)) if valid_teacher_winrates else float('-inf')
                teacher_threshold = min(0.0, teacher_pct)

            logger.info(f"Auto-calculated student threshold: {student_threshold:.3f} (percentile was {student_pct:.3f})")
            logger.info(f"Auto-calculated teacher threshold: {teacher_threshold:.3f} (percentile was {teacher_pct:.3f})")

            # Filter candidates based on thresholds (keep those with valid winrates)
            candidates = []
            for new_candidate in all_candidates:
                # Filter out candidates with None winrates
                if new_candidate[2] is None or new_candidate[3] is None:
                    continue
                # Filter based on student threshold
                if self.direction == "plus":
                    if new_candidate[2] < student_threshold:
                        continue
                    if new_candidate[3] > teacher_threshold:
                        continue
                else:
                    if new_candidate[2] > student_threshold:
                        continue
                    if new_candidate[3] < teacher_threshold:
                        continue
                candidates.append(new_candidate)

            print(
                "===============\n"
                f"After filtering, {len(candidates)} candidates remain."
            )

            # Cluster all filtered candidates upfront for diversity
            candidate_to_cluster: dict[int, int] = {}
            if candidates:
                _, cluster_indices = self.cluster_model.cluster_dbscan(
                    [cand[0] for cand in candidates]
                )
                for label, member_indices in cluster_indices.items():
                    for idx in member_indices:
                        candidate_to_cluster[id(candidates[idx])] = label

                n_clusters = len([k for k in cluster_indices.keys() if k != -1])
                n_outliers = len(cluster_indices.get(-1, []))
                logger.info(f"Clustered {len(candidates)} candidates into {n_clusters} clusters + {n_outliers} outliers.")

            # Sort into Pareto Fronts
            fronts = EvoPlanner._get_pareto_fronts(candidates)

            # Select candidates, enforcing diversity: only one representative per cluster
            final_selection = []
            used_clusters: set[int] = set()

            for front_idx, front in enumerate(fronts):
                if len(final_selection) >= n_pop_target:
                    break

                # Sort front by tiebreak (best first)
                front_sorted = sorted(front, key=EvoPlanner._pareto_tiebreak)
                selected_from_front = 0

                for candidate in front_sorted:
                    if len(final_selection) >= n_pop_target:
                        break

                    cluster_label = candidate_to_cluster.get(id(candidate), -1)

                    # Outliers (-1) are always selectable; clustered candidates only if cluster not used
                    if cluster_label == -1 or cluster_label not in used_clusters:
                        final_selection.append(candidate)
                        selected_from_front += 1
                        if cluster_label != -1:
                            used_clusters.add(cluster_label)

                logger.info(f"Front {front_idx}: {len(front)} candidates, selected {selected_from_front} (enforcing diversity).")

            # Store data for plotting (after selection is complete)
            candidates_by_seed[seed_state.index] = {
                "all_candidates": all_candidates,
                "filtered_candidates": candidates,
                "selected_candidates": final_selection,
                "student_threshold": student_threshold,
                "teacher_threshold": teacher_threshold,
                "direction": self.direction,
            }

            # Update state
            seed_state.state = {
                attribute: time_step
                for attribute, time_step, _, _ in final_selection
            }

            logger.info(
                f"Updated Seed {seed_state.index} population to {len(seed_state.state)} members."
            )

        return candidates_by_seed

    def update_pop(self, seed_states: list[SeedState], n_pop_target: int):
        logger.info(f"Trying to update population to {n_pop_target} members.")
        for seed_state in seed_states:
            candidates = []
            for attribute, stats in seed_state.history[-1].items():
                candidates.append((
                    attribute,
                    stats.meta["time_step"],
                    stats.winrate("student"),
                ))

            _, indices = self.cluster_model.cluster_dbscan([cand[0] for cand in candidates])

            # Select the best candidate from each niche
            representatives = []
            for label, member_indices in indices.items():
                if label == -1:
                    # These are noise points; we'll handle them separately
                    continue

                # Sort members of the niche by score and select the top one
                members = [candidates[i] for i in member_indices]
                if self.direction == "plus":
                    best_in_niche = max(members, key=lambda x: x[2])
                else:
                    best_in_niche = min(members, key=lambda x: x[2])

                representatives.append(best_in_niche)
                logger.info(
                    f"Niche {label}: Selected '{best_in_niche[0]}' with score {best_in_niche[2]}"
                )

            # Handle outliers (prompts labeled as -1)
            outliers = [candidates[i] for i in indices[-1]]
            if self.direction == "plus":
                outliers.sort(key=lambda x: x[2], reverse=True)
            else:
                outliers.sort(key=lambda x: x[2], reverse=False)

            # Combine the best from niches and the best outliers
            combined_selection = representatives + outliers
            if self.direction == "plus":
                combined_selection.sort(key=lambda x: x[2], reverse=True)
            else:
                combined_selection.sort(key=lambda x: x[2], reverse=False)
            final_candidates = combined_selection[:n_pop_target]

            seed_state.state = {
                attribute: time_step for attribute, time_step, _ in final_candidates
            }

            logger.info(
                f"Updated Seed {seed_state.index} population to {len(seed_state.state)} members."
            )


class EvoRunner(Runner):
    planner: EvoPlanner  # for type checker

    def __init__(
        self,
        seed_states: list[SeedState],
        planner: EvoPlanner,
        policy_model: GenerationModel,
        bias_evaluator: BiasEvaluator,
        teacher_model: RewardModel,
        m_var: int,
        n_baseline_rollouts: int,
        n_rewrite_rollouts: int,
        n_validate_rollouts: int,
        run_name: str | None = None,
    ):
        super().__init__(
            seed_states=seed_states,
            policy_model=policy_model,
            bias_evaluator=bias_evaluator,
            teacher_model=teacher_model,
            run_name=run_name,
            n_baseline_rollouts=n_baseline_rollouts,
            n_validate_rollouts=n_validate_rollouts,
        )
        self.planner = planner
        self.bias_evaluator = bias_evaluator
        
        self.m_var = m_var
        self.n_rewrite_rollouts = n_rewrite_rollouts

    @property
    def runner_type(self) -> str:
        return "evo"

    async def train_step(
        self, n_pop_target: int, train_batch_size: int, use_pareto_selection: bool = False
    ):
        print(f"[TRAIN STEP {self.step_count}] Writing new system prompts...")
        if self.step_count == 0:
            await self.planner.initial_plan(runner=self)
        else:
            await self.planner.iterate_plan(
                seed_states=self.seed_states,
                m_var=self.m_var,
            )

        evaluate_results = []
        
        for seed_state_idx, seed_state in enumerate(self.seed_states):
            async with self.bias_evaluator as evaluator:
                user_prompts = random.sample(
                    seed_state.cluster.train_prompts,
                    train_batch_size,
                )
                attributes = list(seed_state.history[-1].keys())
                print(f"Seed {seed_state.index}: evaluating {len(attributes)} attributes")
                # references = [
                #     self.get_references(seed_state_idx, att)
                #     for att in attributes
                # ]
                stats = await evaluator.evaluate_attributes(
                    user_prompts=user_prompts,
                    attributes=attributes,
                    references=None,  # disable references
                    baselines=self.baselines,
                    n_rollouts=self.n_rewrite_rollouts,
                )
            evaluate_results.append(stats)

        if use_pareto_selection:
            # Populate teacher_score on rollouts in place
            await self.teacher_model.judge_validation_results(
                validation_results=evaluate_results,
                val_baselines=self.baselines,  # type: ignore
            )

        for seed_state_idx, stats in enumerate(evaluate_results):
            for attribute, rollouts in stats.items():
                self.seed_states[seed_state_idx].history[-1][attribute].rollouts = rollouts

        self.save_attribute_stats(
            direction=self.planner.direction,
            save_dir=self.run_path / f"step_{self.step_count}_stats",
        )

        logger.info(
            f"[TRAIN STEP {self.step_count}] Updating population. Before update: {len(self.seed_states[0].history[-1])}"
        )
        
        if use_pareto_selection:
            candidates_by_seed = self.planner.update_pop_pareto(
                seed_states=self.seed_states,
                n_pop_target=n_pop_target,
            )

            for seed_state_idx, candidates in candidates_by_seed.items():
                with open(self.run_path / f"step_{self.step_count}_stats" / f"seed_{seed_state_idx}_candidates.json", "w") as f:
                    json.dump([
                        {
                            "attribute": c[0],
                            "student_winrate": c[2],
                            "teacher_winrate": c[3],
                            "time_step": c[1],
                        } 
                        for c in candidates["all_candidates"]
                    ], f, indent=4)

                fig = EvoPlanner.plot_candidate_stats(
                    all_candidates=candidates["all_candidates"],
                    filtered_candidates=candidates["filtered_candidates"],
                    selected_candidates=candidates["selected_candidates"],
                    student_threshold=candidates["student_threshold"],
                    teacher_threshold=candidates["teacher_threshold"],
                    direction=candidates["direction"],
                )
                fig.savefig(self.run_path / f"step_{self.step_count}_stats" / f"seed_{seed_state_idx}_pareto.pdf")


        else:
            self.planner.update_pop(
                seed_states=self.seed_states,
                n_pop_target=n_pop_target,
            )

        logger.info(
            f"[TRAIN STEP {self.step_count}] finished; Current population: {len(self.seed_states[0].state)}"
        )
        self.step_count += 1
        self.planner.hypothesis_planner.step_planner_model()


    async def train(self, n_pop_target: list[int], train_batch_size: list[int], use_pareto_selection: bool = False, validate: bool = True, start_from: int|None=None):
        t_steps = len(train_batch_size)
        assert len(n_pop_target) == t_steps

        for time_step in range(t_steps):
            # if start_from is not None and time_step < start_from:
            #     for seed_state in self.seed_states:
            #         with open(self.run_path / f"step_{time_step}_stats/seed_{seed_state.index}.json", "r") as f:
            #             seed_results = json.load(f)

            #         seed_state.history.append(dict())
            #         for item in seed_results:
            #             attribute = item["attribute"]
            #             attribute_rollouts = dict()
            #             for user_prompt, rollouts in item["all_rollouts"].items():
            #                 attribute_rollouts[user_prompt] = [
            #                     Rollout(
            #                         response=rollout["response"],
            #                         score=rollout["score"]
            #                     )
            #                     if rollout is not None else None
            #                     for rollout in rollouts
            #                 ]

            #             seed_state.history[-1][attribute] = AttributeStats(
            #                 attribute=attribute,
            #                 rollouts=attribute_rollouts,
            #                 meta=item.get("meta", {"time_step": time_step})
            #             )

            #     self.planner.update_pop(
            #         baselines=self.baselines,  # type: ignore
            #         seed_states=self.seed_states,
            #         n_pop_target=n_pop_target[time_step],
            #         dbscan_eps=self.dbscan_eps,
            #     )
            #     self.step_count += 1
            #     self.planner.hypothesis_planner.step_planner_model()
            #     self.save_attribute_stats(
            #         direction=self.planner.direction,
            #         save_dir=None,
            #     )

            # else:
            await self.train_step(
                n_pop_target=n_pop_target[time_step],
                train_batch_size=train_batch_size[time_step],
                use_pareto_selection=use_pareto_selection,
            )

            if validate and time_step == t_steps - 1:
                await self.validate(final_attributes={
                    seed_state.index: list(seed_state.state.keys()) for seed_state in self.seed_states
                })
