# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Type
from pydantic import BaseModel, PrivateAttr, field_validator
from .base_node import BaseNode


class MCTSNode(BaseNode):

    c_puct: float = 2
    inited: bool = False

    __visit_count: int = PrivateAttr(default=0)
    __value_sum: float = PrivateAttr(default=0)

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update_visit_count(self, count: int) -> None:
        self.__visit_count = count

    def update(self, value: float) -> None:
        if self.inited is False:
            self.inited = True
            self.value = value
        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value: float, start_node: Type[BaseNode]) -> None:
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def puct(self) -> float:
        if not self.parent: return 0
        q_value = self.q_value() if self.visit_count() > 0 else 0
        # AlphaZero-style exploration bonus: scales with sqrt(parent visits)
        # and decays as this child accumulates its own visits, so it stays
        # well-defined -- and strictly positive whenever the parent has been
        # visited -- for an as-yet-unvisited child. The previous formula
        # hardcoded this term to 0 whenever visit_count()==0, so a freshly
        # created child got no credit at all for being unexplored: once any
        # one sibling's q_value turned positive, it permanently outranked
        # every unvisited sibling (puct()==0) and the search never revisited
        # them again. See methodology.tex Sec. mcts_search.
        u_value = self.c_puct * np.sqrt(self.parent.visit_count()) / (1 + self.visit_count())
        return q_value + u_value
        
