import unittest
from akv_cache_engine import CacheAwareContextManager

class TestAKVCacheEngine(unittest.TestCase):
    def setUp(self):
        self.mgr = CacheAwareContextManager(ttl_seconds=3600)

    def test_static_prefix_pinning_and_hit(self):
        sys_prompt = "You are a senior operations engineer."
        tools = "bash, read_file"
        workspace = "ARM64 12GB RAM"
        history = [{"role": "user", "content": "hello"}]

        # Turn 1: Miss
        res1 = self.mgr.build_structured_context(sys_prompt, tools, workspace, history)
        self.assertFalse(res1["cache_hit"])
        self.assertGreater(res1["static_tokens"], 0)

        # Turn 2: Exact Prefix Match -> Hit
        res2 = self.mgr.build_structured_context(sys_prompt, tools, workspace, history)
        self.assertTrue(res2["cache_hit"])
        self.assertEqual(res1["static_prefix_hash"], res2["static_prefix_hash"])

    def test_rolling_drop_eviction(self):
        history = [
            {"role": "user", "content": "Run tool"},
            {"role": "tool", "content": "ephemeral log dump " * 500, "ephemeral": True},
            {"role": "assistant", "content": "Tool finished"},
            {"role": "user", "content": "Next turn"}
        ]
        res = self.mgr.build_structured_context("sys", "tools", "ws", history, rolling_drop_policy=True)
        self.assertGreater(res["dropped_tokens"], 1000)
        self.assertEqual(res["dropped_turns"], 1)

    def test_financial_savings_calculation(self):
        savings = self.mgr.calculate_cost_savings(
            unoptimized_tokens=100000,
            cached_tokens=80000,
            eval_tokens=20000,
            price_per_million_input=2.50,
            cached_discount=0.90
        )
        self.assertGreater(savings["savings_percentage"], 65.0)

if __name__ == "__main__":
    unittest.main()
