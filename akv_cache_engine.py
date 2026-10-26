#!/usr/bin/env python3
"""
Cache-Aware Agent Engine (AKV Emulation & Prompt Cache Optimization)
Implements:
1. Static Prefix Pinning & Boundary Isolation (System Prompts, Tools, Knowledge)
2. Programmatic Cache Control (drop & repos) for Multi-Turn Agent Loops
3. Token and Latency Accounting with Monetization Savings Analytics
"""

import hashlib
import time
import json
from typing import List, Dict, Any, Optional

class CacheAwareContextManager:
    """
    Manages prompt composition and KV cache simulation/attribution
    for autonomous agent loops using AKV principles.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        # In-memory simulated KV cache prefix registry: prefix_hash -> metadata
        self.kv_registry: Dict[str, Dict[str, Any]] = {}

    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.strip().encode('utf-8')).hexdigest()

    def build_structured_context(
        self,
        system_instructions: str,
        tool_schemas: str,
        workspace_context: str,
        turn_history: List[Dict[str, str]],
        rolling_drop_policy: bool = True
    ) -> Dict[str, Any]:
        """
        Partition prompt into:
        - [STATIC CACHED PREFIX]: System Prompt + Tools + Workspace Guidelines.
          This prefix MUST NEVER mutate between agent steps to guarantee 100% prefix cache hits.
        - [DYNAMIC EPHEMERAL TAIL]: Rolling agent-tool interactions managed by 'drop'.
        """
        static_block = (
            f"<SYSTEM_RULES>\n{system_instructions}\n</SYSTEM_RULES>\n"
            f"<AVAILABLE_TOOLS>\n{tool_schemas}\n</AVAILABLE_TOOLS>\n"
            f"<WORKSPACE_GUIDELINES>\n{workspace_context}\n</WORKSPACE_GUIDELINES>\n"
        )
        static_hash = self.compute_hash(static_block)
        now = time.time()

        # Check KV Cache Status
        cache_hit = False
        if static_hash in self.kv_registry:
            entry = self.kv_registry[static_hash]
            if now - entry["created_at"] < self.ttl_seconds:
                cache_hit = True
                entry["hits"] += 1
                entry["last_accessed"] = now

        if not cache_hit:
            self.kv_registry[static_hash] = {
                "created_at": now,
                "last_accessed": now,
                "hits": 0,
                "token_estimate": len(static_block) // 4
            }

        # Apply AKV Rolling Drop to Turn History
        active_turns = []
        dropped_turns = 0
        dropped_tokens = 0

        for idx, turn in enumerate(turn_history):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            is_intermediate_tool = turn.get("ephemeral", False) or (
                role == "tool" and idx < len(turn_history) - 2
            )

            if rolling_drop_policy and is_intermediate_tool:
                # Simulates the AKV 'drop' operation:
                # Exclude old verbose tool payload from active attention
                dropped_turns += 1
                dropped_tokens += len(content) // 4
                # Retain compact pointer/summary to keep causal reasoning intact without bloated tokens
                active_turns.append({
                    "role": "tool_reference",
                    "content": f"[CACHE_ENGINEERING: Tool output at step {idx} dropped from active KV memory]",
                    "status": "dropped"
                })
            else:
                active_turns.append(turn)

        static_tokens = len(static_block) // 4
        dynamic_tokens = sum(len(t.get("content", "")) // 4 for t in active_turns)
        total_tokens = static_tokens + dynamic_tokens

        return {
            "static_prefix_hash": static_hash,
            "cache_hit": cache_hit,
            "static_tokens": static_tokens,
            "dynamic_tokens": dynamic_tokens,
            "total_tokens": total_tokens,
            "dropped_tokens": dropped_tokens,
            "dropped_turns": dropped_turns,
            "active_turns": active_turns,
            "cache_savings_ratio": round((dropped_tokens + (static_tokens if cache_hit else 0)) / (total_tokens + dropped_tokens or 1), 4)
        }

    def calculate_cost_savings(
        self,
        unoptimized_tokens: int,
        cached_tokens: int,
        eval_tokens: int,
        price_per_million_input: float = 2.50,
        cached_discount: float = 0.90 # 90% discount on cached tokens (e.g. Gemini / Anthropic cache pricing)
    ) -> Dict[str, float]:
        """
        Quantify financial savings from cache hits and dropped tokens.
        """
        baseline_cost = (unoptimized_tokens / 1_000_000) * price_per_million_input
        optimized_cost = (
            (eval_tokens / 1_000_000) * price_per_million_input +
            (cached_tokens / 1_000_000) * (price_per_million_input * (1.0 - cached_discount))
        )
        savings = max(0.0, baseline_cost - optimized_cost)
        percent_saved = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return {
            "baseline_cost_usd": round(baseline_cost, 6),
            "optimized_cost_usd": round(optimized_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_percentage": round(percent_saved, 2)
        }
