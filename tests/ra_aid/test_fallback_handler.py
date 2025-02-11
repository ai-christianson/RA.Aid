import unittest
from ra_aid.fallback_handler import FallbackHandler

class DummyLogger:
    def debug(self, msg):
        pass
    def error(self, msg):
        pass

class DummyAgent:
    provider = "openai"
    tools = []
    model = None

class TestFallbackHandler(unittest.TestCase):
    def setUp(self):
        self.config = {"max_tool_failures": 2, "fallback_tool_models": "dummy-fallback-model"}
        self.fallback_handler = FallbackHandler(self.config)
        self.logger = DummyLogger()
        self.agent = DummyAgent()

    def test_handle_failure_increments_counter(self):
        initial_failures = self.fallback_handler.tool_failure_consecutive_failures
        self.fallback_handler.handle_failure("dummy_call()", Exception("Test error"), self.logger, self.agent)
        self.assertEqual(self.fallback_handler.tool_failure_consecutive_failures, initial_failures + 1)

    def test_attempt_fallback_resets_counter(self):
        # Monkey-patch dummy functions for fallback components
        def dummy_initialize_llm(provider, model_name, temperature=None):
            class DummyModel:
                def bind_tools(self, tools, tool_choice):
                    pass
            return DummyModel()

        def dummy_merge_chat_history():
            return ["merged"]

        def dummy_validate_provider_env(provider):
            return True

        import ra_aid.llm as llm
        original_initialize = llm.initialize_llm
        original_merge = llm.merge_chat_history
        original_validate = llm.validate_provider_env
        llm.initialize_llm = dummy_initialize_llm
        llm.merge_chat_history = dummy_merge_chat_history
        llm.validate_provider_env = dummy_validate_provider_env

        self.fallback_handler.tool_failure_consecutive_failures = 2
        self.fallback_handler.attempt_fallback("dummy_tool_call()", self.logger, self.agent)
        self.assertEqual(self.fallback_handler.tool_failure_consecutive_failures, 0)

        llm.initialize_llm = original_initialize
        llm.merge_chat_history = original_merge
        llm.validate_provider_env = original_validate

if __name__ == "__main__":
    unittest.main()
