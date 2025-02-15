import unittest

from ra_aid.exceptions import FallbackToolExecutionError
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
        self.config = {
            "max_tool_failures": 2,
            "fallback_tool_models": "dummy-fallback-model",
            "experimental_fallback_handler": True,
        }
        self.fallback_handler = FallbackHandler(self.config, [])
        self.logger = DummyLogger()
        self.agent = DummyAgent()

        def dummy_tool():
            pass

        class DummyToolWrapper:
            def __init__(self, func):
                self.func = func

        self.agent.tools = [DummyToolWrapper(dummy_tool)]

    def test_handle_failure_increments_counter(self):
        from ra_aid.exceptions import ToolExecutionError

        initial_failures = self.fallback_handler.tool_failure_consecutive_failures
        error_obj = ToolExecutionError(
            "Test error", base_message="dummy_call()", tool_name="dummy_tool"
        )
        self.fallback_handler.handle_failure(error_obj, self.agent, [])
        self.assertEqual(
            self.fallback_handler.tool_failure_consecutive_failures,
            initial_failures + 1,
        )

    def test_attempt_fallback_resets_counter(self):
        # Monkey-patch dummy functions for fallback components
        def dummy_initialize_llm(provider, model_name, temperature=None):
            class DummyModel:
                def bind_tools(self, tools, tool_choice):
                    pass

            return DummyModel()

        def dummy_validate_provider_env(provider):
            return True

        import ra_aid.llm as llm

        original_initialize = llm.initialize_llm
        original_validate = llm.validate_provider_env
        llm.initialize_llm = dummy_initialize_llm
        llm.validate_provider_env = dummy_validate_provider_env

        self.fallback_handler.tool_failure_consecutive_failures = 2
        with self.assertRaises(FallbackToolExecutionError):
            self.fallback_handler.attempt_fallback()
        self.assertEqual(self.fallback_handler.tool_failure_consecutive_failures, 0)

        llm.initialize_llm = original_initialize
        llm.validate_provider_env = original_validate

    def test_load_fallback_tool_models(self):
        import ra_aid.fallback_handler as fh

        original_supported = fh.supported_top_tool_models
        fh.supported_top_tool_models = [
            {"provider": "dummy", "model": "dummy_model", "type": "prompt"}
        ]
        models = self.fallback_handler._load_fallback_tool_models(self.config)
        self.assertIsInstance(models, list)
        fh.supported_top_tool_models = original_supported

    def test_extract_failed_tool_name(self):
        from ra_aid.exceptions import FallbackToolExecutionError, ToolExecutionError

        # Case when tool_name is provided
        error1 = ToolExecutionError(
            "Error", base_message="dummy", tool_name="dummy_tool"
        )
        name1 = self.fallback_handler.extract_failed_tool_name(error1)
        self.assertEqual(name1, "dummy_tool")
        # Case when tool_name is not provided but regex works
        error2 = ToolExecutionError('error with name="test_tool"')
        name2 = self.fallback_handler.extract_failed_tool_name(error2)
        self.assertEqual(name2, "test_tool")
        # Case when regex fails and exception is raised
        error3 = ToolExecutionError("no tool name here")
        with self.assertRaises(FallbackToolExecutionError):
            self.fallback_handler.extract_failed_tool_name(error3)

    def test_find_tool_to_bind(self):
        class DummyWrapper:
            def __init__(self, func):
                self.func = func

        def dummy_func(_args):
            return "result"

        dummy_wrapper = DummyWrapper(dummy_func)
        self.agent.tools.append(dummy_wrapper)
        tool = self.fallback_handler._find_tool_to_bind(self.agent, dummy_func.__name__)
        self.assertIsNotNone(tool)
        self.assertEqual(tool.func.__name__, dummy_func.__name__)

    def test_bind_tool_model(self):
        # Setup a dummy simple_model with bind_tools method
        class DummyModel:
            def bind_tools(self, tools, tool_choice=None):
                self.bound = True
                self.tools = tools
                self.tool_choice = tool_choice
                return self

            def with_retry(self, stop_after_attempt):
                return self

            def invoke(self, msg_list):
                return "dummy_response"

        dummy_model = DummyModel()

        # Set current tool for binding
        class DummyTool:
            def invoke(self, args):
                return "result"

        self.fallback_handler.current_tool_to_bind = DummyTool()
        self.fallback_handler.current_failing_tool_name = "test_tool"
        # Test with force calling ("fc") type
        fallback_model_fc = {"type": "fc"}
        bound_model_fc = self.fallback_handler._bind_tool_model(
            dummy_model, fallback_model_fc
        )
        self.assertTrue(hasattr(bound_model_fc, "tool_choice"))
        self.assertEqual(bound_model_fc.tool_choice, "test_tool")
        # Test with prompt type
        fallback_model_prompt = {"type": "prompt"}
        bound_model_prompt = self.fallback_handler._bind_tool_model(
            dummy_model, fallback_model_prompt
        )
        self.assertTrue(bound_model_prompt.tool_choice is None)

    def test_invoke_fallback(self):
        import os
        from unittest.mock import patch

        # Successful fallback scenario with proper API key set
        with (
            patch.dict(os.environ, {"DUMMY_API_KEY": "dummy_value"}),
            patch(
                "ra_aid.fallback_handler.supported_top_tool_models",
                new=[{"provider": "dummy", "model": "dummy_model", "type": "prompt"}],
            ),
            patch("ra_aid.fallback_handler.validate_provider_env", return_value=True),
            patch("ra_aid.fallback_handler.initialize_llm") as mock_init_llm,
        ):

            class DummyModel:
                def bind_tools(self, tools, tool_choice=None):
                    return self

                def with_retry(self, stop_after_attempt):
                    return self

                def invoke(self, msg_list):
                    return DummyResponse()

            class DummyResponse:
                additional_kwargs = {
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "test",
                            "function": {"name": "dummy_tool", "arguments": '{"a":1}'},
                        }
                    ]
                }

            def dummy_initialize_llm(provider, model_name):
                return DummyModel()

            mock_init_llm.side_effect = dummy_initialize_llm

            # Set current tool for fallback
            class DummyTool:
                def invoke(self, args):
                    return "tool_result"

            self.fallback_handler.current_tool_to_bind = DummyTool()
            self.fallback_handler.current_failing_tool_name = "dummy_tool"
            # Add dummy tool for lookup in invoke_prompt_tool_call
            self.fallback_handler.tools.append(
                type(
                    "DummyToolWrapper",
                    (),
                    {
                        "func": type("DummyToolFunc", (), {"__name__": "dummy_tool"})(),
                        "invoke": lambda self, args=None: "tool_result",
                    },
                )
            )
            result = self.fallback_handler.invoke_fallback(
                {"provider": "dummy", "model": "dummy_model", "type": "prompt"}
            )
            self.assertIsInstance(result, list)
            self.assertEqual(result[1], "tool_result")

        # Failed fallback scenario due to missing API key (simulate by empty environment)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ra_aid.fallback_handler.supported_top_tool_models",
                new=[{"provider": "dummy", "model": "dummy_model", "type": "prompt"}],
            ),
            patch("ra_aid.fallback_handler.validate_provider_env", return_value=False),
            patch("ra_aid.fallback_handler.initialize_llm") as mock_init_llm,
        ):

            class FailingDummyModel:
                def bind_tools(self, tools, tool_choice=None):
                    return self

                def with_retry(self, stop_after_attempt):
                    return self

                def invoke(self, msg_list):
                    raise Exception("API key missing")

            def failing_initialize_llm(provider, model_name):
                return FailingDummyModel()

            mock_init_llm.side_effect = failing_initialize_llm
            fallback_result = self.fallback_handler.invoke_fallback(
                {"provider": "dummy", "model": "dummy_model", "type": "prompt"}
            )
            self.assertIsNone(fallback_result)

        # Test that the overall fallback mechanism raises FallbackToolExecutionError when all models fail
        # Set failure count to trigger the fallback attempt in attempt_fallback
        from ra_aid.exceptions import FallbackToolExecutionError

        self.fallback_handler.tool_failure_consecutive_failures = (
            self.fallback_handler.max_failures
        )
        with self.assertRaises(FallbackToolExecutionError) as cm:
            self.fallback_handler.attempt_fallback()
        self.assertIn("All fallback models have failed", str(cm.exception))

    def test_construct_prompt_msg_list(self):
        msgs = self.fallback_handler.construct_prompt_msg_list()
        from ra_aid.fallback_handler import HumanMessage, SystemMessage

        self.assertTrue(any(isinstance(m, SystemMessage) for m in msgs))
        self.assertTrue(any(isinstance(m, HumanMessage) for m in msgs))
        # Test with failed_messages added
        self.fallback_handler.failed_messages.append("failed_msg")
        msgs_with_fail = self.fallback_handler.construct_prompt_msg_list()
        self.assertTrue(any("failed_msg" in str(m) for m in msgs_with_fail))

    def test_invoke_prompt_tool_call(self):
        # Create dummy tool function
        def dummy_tool_func(args):
            return "invoked_result"

        dummy_tool_func.__name__ = "dummy_tool"

        # Create wrapper class
        class DummyToolWrapper:
            def __init__(self, func):
                self.func = func

            def invoke(self, args):
                return self.func(args)

        dummy_wrapper = DummyToolWrapper(dummy_tool_func)
        self.fallback_handler.tools = [dummy_wrapper]
        tool_call_req = {"name": "dummy_tool", "arguments": {"x": 42}}
        result = self.fallback_handler.invoke_prompt_tool_call(tool_call_req)
        self.assertEqual(result, "invoked_result")

    def test_base_message_to_tool_call_dict(self):
        dummy_tool_call = {
            "id": "123",
            "type": "test",
            "function": {"name": "dummy_tool", "arguments": '{"x":42}'},
        }
        DummyResponse = type(
            "DummyResponse",
            (),
            {"additional_kwargs": {"tool_calls": [dummy_tool_call]}},
        )
        result = self.fallback_handler.base_message_to_tool_call_dict(DummyResponse)
        self.assertEqual(result["id"], "123")
        self.assertEqual(result["name"], "dummy_tool")
        self.assertEqual(result["arguments"], {"x": 42})

    def test_parse_tool_arguments(self):
        args_str = '{"a": 1}'
        parsed = self.fallback_handler._parse_tool_arguments(args_str)
        self.assertEqual(parsed, {"a": 1})
        args_dict = {"b": 2}
        parsed_dict = self.fallback_handler._parse_tool_arguments(args_dict)
        self.assertEqual(parsed_dict, {"b": 2})

    def test_get_tool_calls(self):
        DummyResponse = type("DummyResponse", (), {})()
        DummyResponse.additional_kwargs = {"tool_calls": [{"id": "1"}]}
        calls = self.fallback_handler.get_tool_calls(DummyResponse)
        self.assertEqual(calls, [{"id": "1"}])
        DummyResponse2 = type("DummyResponse2", (), {"tool_calls": [{"id": "2"}]})()
        calls2 = self.fallback_handler.get_tool_calls(DummyResponse2)
        self.assertEqual(calls2, [{"id": "2"}])
        dummy_dict = {"additional_kwargs": {"tool_calls": [{"id": "3"}]}}
        calls3 = self.fallback_handler.get_tool_calls(dummy_dict)
        self.assertEqual(calls3, [{"id": "3"}])

    def test_handle_failure_response(self):
        from ra_aid.exceptions import ToolExecutionError

        def dummy_handle_failure(error, agent):
            return ["fallback_response"]

        self.fallback_handler.handle_failure = dummy_handle_failure
        response = self.fallback_handler.handle_failure_response(
            ToolExecutionError("test", tool_name="dummy_tool"), self.agent, "React"
        )
        from ra_aid.fallback_handler import SystemMessage

        self.assertTrue(all(isinstance(m, SystemMessage) for m in response))
        response_non = self.fallback_handler.handle_failure_response(
            ToolExecutionError("test", tool_name="dummy_tool"), self.agent, "Other"
        )
        self.assertIsNone(response_non)

    def test_init_msg_list_non_overlapping(self):
        # Test when the first two and last two messages do not overlap.
        full_list = ["msg1", "msg2", "msg3", "msg4", "msg5"]
        self.fallback_handler.init_msg_list(full_list)
        # Expected merged list: first two ("msg1", "msg2") plus last two ("msg4", "msg5")
        self.assertEqual(
            self.fallback_handler.msg_list, ["msg1", "msg2", "msg4", "msg5"]
        )

    def test_init_msg_list_with_overlap(self):
        # Test when the last two messages overlap with the first two.
        full_list = ["msg1", "msg2", "msg1", "msg3"]
        self.fallback_handler.init_msg_list(full_list)
        # Expected merged list: first two ("msg1", "msg2") plus "msg3" from the last two, since "msg1" was already present.
        self.assertEqual(self.fallback_handler.msg_list, ["msg1", "msg2", "msg3"])


if __name__ == "__main__":
    unittest.main()
