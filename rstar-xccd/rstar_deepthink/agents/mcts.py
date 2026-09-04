# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import re
from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import field_validator
from vllm.outputs import CompletionOutput, RequestOutput
from rstar_deepthink.agents.utils import math_equiv as is_equiv
from rstar_deepthink.nodes.base_node import BaseNode
from rstar_deepthink.nodes import MCTSNode
from rstar_deepthink.constants import (
    TOO_MANY_CODE_ERRORS,
    TOO_MANY_STEPS,
    NO_VALID_CHILD,
    CODE_END,
)
from .tree import BaseTree, code_execution, collect_action_inputs, extract_program
from .beam_search import BS
from evaluate_clone_results import normalize_label

# Matches an `assert` statement in generated code, used to decide whether a
# python_interpreter step made an equivalence claim worth checking for
# self-consistency against the branch's eventual conclusion (see
# create_child() and MCTS._score_pending_assert_verdicts()).
ASSERT_RE = re.compile(r"\bassert\b")


class MCTS(BS):
    search_node: Type[BaseNode] = None

    intermediate_metric: Dict = {
        "question": "",
        "gt": "", 
        "answers": [],
        "judgements": [],
        "value_estimate": [],
        "rollout_indexs": [],
    }

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.NODE_KEYS,
            c_puct=self.config.c_puct,
        )


    def selection(self, from_root=False) -> Optional[Type[MCTSNode]]:
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node
        # select a child node
        node = start_node
        if node is None: return None
        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True
            node = next_node
        return None if (node is None or node.is_terminal) else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        #return random.choice(best_childs) if best_childs else None
        return best_childs[0] if best_childs else None

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        for idx, output in enumerate(outputs):
            if not isinstance(output.stop_reason, str): output.stop_reason = ""
            step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
            self.create_child(step_result, parser_result, node, idx)

    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        idx: int,
    ) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            else:
                new_node.state["text"] = step_result
                
            code_ran_ok = "error" not in observation.lower()
            is_assertion_error = observation.startswith("AssertionError")
            # A completed self-check (assertion held or failed) is not a
            # code defect -- don't count it toward errors_threshold. Only a
            # genuine crash (any other exception, or unparsable input) does.
            if not code_ran_ok and not is_assertion_error:
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors >= self.config.errors_threshold:
                    observation = self.obs_wrap(observation)
                    step_result = step_result + CODE_END if CODE_END not in step_result else step_result
                    new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)

            if not new_node.is_terminal:
                if is_assertion_error:
                    # The branch ran its own equivalence check and found the
                    # snippets differ. Don't reward/penalize this outright --
                    # whether that was the "right" outcome depends on the
                    # branch's eventual conclusion, which doesn't exist yet.
                    # Scored retroactively in eval_final_answer() instead, so
                    # the signal is about self-consistency (does the
                    # conclusion follow from this branch's own evidence),
                    # never about ground truth.
                    new_node.state["assert_verdict"] = "non-clone"
                elif not code_ran_ok:
                    # Genuine crash, independent of is_sampling/ground
                    # truth: PUCT should favor paths with valid, executable
                    # Python over ones that error out.
                    new_node.update_recursive(self.config.negative_reward, self.root)
                else:
                    history_action_inputs = collect_action_inputs(node, parser_result["action"])
                    executed_code = extract_program(''.join(history_action_inputs) + parser_result["action_input"])
                    if ASSERT_RE.search(executed_code):
                        # Ran cleanly *and* the code made an equivalence
                        # claim (an assert that held). Same deferred
                        # treatment as the AssertionError case above, with
                        # the opposite implied verdict.
                        new_node.state["assert_verdict"] = "clone"
                    else:
                        # Plain clean execution with nothing to verify
                        # against the eventual conclusion -- score
                        # immediately, as before.
                        new_node.update_recursive(self.config.positive_reward, self.root)
        else:
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            # if the final answer is not valid, update the node with negative reward
            node.update(self.config.negative_reward)
            return

        self._score_pending_assert_verdicts(node)

        if self.config.is_sampling:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            # just append the node to candidate_nodes, will update the value in select_next_step()
            self.candidate_nodes.append(node)

    def _score_pending_assert_verdicts(self, node: Type[MCTSNode]) -> None:
        """Retroactively score ancestor code-execution steps that ran an
        equivalence assertion (see create_child()) against this leaf's
        actual conclusion.

        create_child() defers scoring a python_interpreter step the moment
        it hits a completed assert (pass or AssertionError), because the
        branch's eventual conclusion doesn't exist yet -- rewarding/
        penalizing it immediately would mean guessing, or worse, leaking
        which answer is "right". Once a leaf resolves to a real answer, we
        walk back up and check whether each ancestor's assertion outcome
        actually agrees with what the branch concluded: reward
        self-consistency, penalize contradicting your own evidence (e.g.
        the assertion found the snippets differ, but the branch still
        concludes "clone" some other way). This never touches
        self.ground_truth -- it only compares the branch's evidence to its
        own conclusion, so it's meaningful regardless of is_sampling.

        A single ancestor can be visited by multiple descendant leaves
        across different rollouts; each one contributes its own
        consistency judgment via update_recursive's running average, same
        as every other reward in this tree.
        """
        leaf_label = normalize_label(node.state.get("final_answer", ""))
        if leaf_label is None:
            return
        ancestor = node.parent
        while ancestor is not None:
            verdict = ancestor.state.get("assert_verdict")
            if verdict is not None:
                reward = self.config.positive_reward if verdict == leaf_label else self.config.negative_reward
                ancestor.update_recursive(reward, self.root)
            ancestor = ancestor.parent

    def record_intermediate_metric(self, answer, value_estimate):
        self.intermediate_metric["question"] = self.question
        self.intermediate_metric["gt"] = self.ground_truth
        # each rollout retains the answer with the highest value_estimate
        # Check if the rollout's answer is already in the list
        if self.intermediate_metric["rollout_indexs"] and self.rollout_idx in self.intermediate_metric["rollout_indexs"]:
            # Find the index of the existing rollout
            index = self.intermediate_metric["rollout_indexs"].index(self.rollout_idx)
            if value_estimate > self.intermediate_metric["value_estimate"][index]:
                self.intermediate_metric["answers"][index] = answer
                self.intermediate_metric["judgements"][index] = is_equiv(self.ground_truth, answer)
                self.intermediate_metric["value_estimate"][index] = value_estimate
        else:
            # If the rollout's answer is not in the list, add it
            self.intermediate_metric["answers"].append(answer)
            self.intermediate_metric["judgements"].append(is_equiv(self.ground_truth, answer))
            self.intermediate_metric["value_estimate"].append(value_estimate)
            self.intermediate_metric["rollout_indexs"].append(self.rollout_idx)


    def select_next_step(self, outputs=None, from_root=False) -> None:
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []
        if outputs:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and self.config.is_sampling:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                # backup
                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    # for terminal node: update_recursive
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        # save intermediate metric
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"], value_estimate=value_estimate)

                        candidate_node.update_recursive(value_estimate, self.root)
                else:
                    # for intermediate node: just update the value
                    if self.config.terminal_sample:
                        pass
                    else:
                        candidate_node.update(value_estimate)

                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection(from_root=from_root)
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            value_estimate = output.value_estimate
            if value_estimate is not None:  
                self.expand_node(output.outputs, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True

            if self.config.update_leaf_value:
                # if need update leaf node value, just append the node to candidate_nodes, will update the value in select_next_step()
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node) 
                    
                    
    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states
