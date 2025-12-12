"""Tests for hark.utils module."""

import io
import os
import sys

import pytest

from hark.utils import env_vars, renumber_speaker, suppress_output


class TestSuppressOutput:
    """Tests for suppress_output context manager."""

    def test_suppresses_stdout(self) -> None:
        """Test that stdout is suppressed within context."""
        original_stdout = sys.stdout

        with suppress_output():
            print("This should not appear")
            # Verify stdout is redirected
            assert sys.stdout is not original_stdout

        # After context, stdout is restored
        assert sys.stdout is original_stdout

    def test_suppresses_stderr(self) -> None:
        """Test that stderr is suppressed within context."""
        original_stderr = sys.stderr

        with suppress_output():
            print("Error message", file=sys.stderr)
            # Verify stderr is redirected
            assert sys.stderr is not original_stderr

        # After context, stderr is restored
        assert sys.stderr is original_stderr

    def test_restores_streams_on_success(self) -> None:
        """Test that streams are restored after successful execution."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with suppress_output():
            pass

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_restores_streams_on_exception(self) -> None:
        """Test that streams are restored even when exception occurs."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with pytest.raises(ValueError):
            with suppress_output():
                raise ValueError("test error")

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_nested_suppress_output(self) -> None:
        """Test that nested suppress_output contexts work correctly."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with suppress_output():
            inner_stdout = sys.stdout
            inner_stderr = sys.stderr

            with suppress_output():
                # Inner context should have different StringIO
                assert sys.stdout is not inner_stdout
                assert sys.stderr is not inner_stderr

            # After inner context, streams are restored to inner's
            assert sys.stdout is inner_stdout
            assert sys.stderr is inner_stderr

        # After outer context, original streams are restored
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_output_not_visible_outside(self) -> None:
        """Test that suppressed output is not visible."""
        captured_output = io.StringIO()
        sys.stdout = captured_output

        with suppress_output():
            print("Hidden message")

        # Restore original stdout before assertion
        sys.stdout = sys.__stdout__

        # The hidden message should not appear in captured output
        assert "Hidden message" not in captured_output.getvalue()


class TestRenumberSpeaker:
    """Tests for renumber_speaker function."""

    def test_renumber_speaker_00_default_offset(self) -> None:
        """Test SPEAKER_00 becomes SPEAKER_01 with default offset."""
        assert renumber_speaker("SPEAKER_00") == "SPEAKER_01"

    def test_renumber_speaker_05_default_offset(self) -> None:
        """Test SPEAKER_05 becomes SPEAKER_06 with default offset."""
        assert renumber_speaker("SPEAKER_05") == "SPEAKER_06"

    def test_renumber_speaker_99_default_offset(self) -> None:
        """Test SPEAKER_99 becomes SPEAKER_100 with default offset."""
        assert renumber_speaker("SPEAKER_99") == "SPEAKER_100"

    def test_renumber_speaker_with_zero_offset(self) -> None:
        """Test speaker label unchanged with offset=0."""
        assert renumber_speaker("SPEAKER_00", offset=0) == "SPEAKER_00"
        assert renumber_speaker("SPEAKER_05", offset=0) == "SPEAKER_05"

    def test_renumber_speaker_with_custom_offset(self) -> None:
        """Test speaker renumbering with custom offset."""
        assert renumber_speaker("SPEAKER_00", offset=5) == "SPEAKER_05"
        assert renumber_speaker("SPEAKER_10", offset=2) == "SPEAKER_12"

    def test_renumber_speaker_negative_offset(self) -> None:
        """Test speaker renumbering with negative offset."""
        assert renumber_speaker("SPEAKER_05", offset=-1) == "SPEAKER_04"
        assert renumber_speaker("SPEAKER_10", offset=-5) == "SPEAKER_05"

    def test_renumber_speaker_non_speaker_prefix(self) -> None:
        """Test non-SPEAKER_ labels are returned unchanged."""
        assert renumber_speaker("UNKNOWN") == "UNKNOWN"
        assert renumber_speaker("John") == "John"
        assert renumber_speaker("") == ""

    def test_renumber_speaker_invalid_format(self) -> None:
        """Test SPEAKER_ labels with invalid format are returned unchanged."""
        # No underscore
        assert renumber_speaker("SPEAKER") == "SPEAKER"
        # Non-numeric suffix
        assert renumber_speaker("SPEAKER_abc") == "SPEAKER_abc"
        assert renumber_speaker("SPEAKER_") == "SPEAKER_"
        # Multiple underscores (takes first part after underscore)
        assert renumber_speaker("SPEAKER_00_extra") == "SPEAKER_01"

    def test_renumber_speaker_preserves_leading_zeros(self) -> None:
        """Test that result maintains 2-digit format with leading zeros."""
        assert renumber_speaker("SPEAKER_00") == "SPEAKER_01"
        assert renumber_speaker("SPEAKER_08") == "SPEAKER_09"
        assert renumber_speaker("SPEAKER_09") == "SPEAKER_10"

    def test_renumber_speaker_single_digit_input(self) -> None:
        """Test handling of single-digit input (if encountered)."""
        # The function handles any numeric suffix
        assert renumber_speaker("SPEAKER_0") == "SPEAKER_01"
        assert renumber_speaker("SPEAKER_5") == "SPEAKER_06"

    def test_renumber_speaker_three_digit_input(self) -> None:
        """Test handling of three-digit input."""
        # Output always uses :02d format, but handles larger numbers
        assert renumber_speaker("SPEAKER_100") == "SPEAKER_101"
        assert renumber_speaker("SPEAKER_999") == "SPEAKER_1000"

    def test_renumber_speaker_case_sensitivity(self) -> None:
        """Test that function is case-sensitive."""
        # Only SPEAKER_ prefix is recognized, not speaker_ or Speaker_
        assert renumber_speaker("speaker_00") == "speaker_00"
        assert renumber_speaker("Speaker_00") == "Speaker_00"

    def test_renumber_speaker_whitespace(self) -> None:
        """Test handling of whitespace in input."""
        # Leading whitespace prevents prefix match
        assert renumber_speaker(" SPEAKER_00") == " SPEAKER_00"
        # Trailing whitespace is stripped by int() conversion
        assert renumber_speaker("SPEAKER_00 ") == "SPEAKER_01"

    def test_renumber_speaker_type_hints(self) -> None:
        """Test that function accepts str and returns str."""
        result = renumber_speaker("SPEAKER_00")
        assert isinstance(result, str)


class TestRenumberSpeakerEdgeCases:
    """Edge case tests for renumber_speaker."""

    def test_empty_string(self) -> None:
        """Test empty string input."""
        assert renumber_speaker("") == ""

    def test_speaker_prefix_only(self) -> None:
        """Test SPEAKER_ with no suffix."""
        assert renumber_speaker("SPEAKER_") == "SPEAKER_"

    def test_very_large_offset(self) -> None:
        """Test with very large offset."""
        assert renumber_speaker("SPEAKER_00", offset=1000) == "SPEAKER_1000"

    def test_offset_causes_negative_result(self) -> None:
        """Test when offset would cause negative result."""
        # Python handles negative numbers; :02d format doesn't pad negatives
        result = renumber_speaker("SPEAKER_00", offset=-5)
        assert result == "SPEAKER_-5"

    def test_speaker_with_leading_zeros_preserved_in_input(self) -> None:
        """Test input with various leading zero patterns."""
        assert renumber_speaker("SPEAKER_007") == "SPEAKER_08"
        assert renumber_speaker("SPEAKER_000") == "SPEAKER_01"


class TestEnvVars:
    """Tests for env_vars context manager."""

    def test_sets_env_var_within_context(self) -> None:
        """Test that env var is set within the context."""
        test_var = "_HARK_TEST_VAR_1"
        # Ensure var doesn't exist
        os.environ.pop(test_var, None)

        with env_vars({test_var: "test_value"}):
            assert os.environ.get(test_var) == "test_value"

        # After context, var should be removed
        assert test_var not in os.environ

    def test_removes_new_var_after_context(self) -> None:
        """Test that new env vars are removed after context exits."""
        test_var = "_HARK_TEST_VAR_2"
        os.environ.pop(test_var, None)

        with env_vars({test_var: "temporary"}):
            pass

        assert test_var not in os.environ

    def test_restores_existing_var_after_context(self) -> None:
        """Test that existing env vars are restored to original value."""
        test_var = "_HARK_TEST_VAR_3"
        original_value = "original"
        os.environ[test_var] = original_value

        try:
            with env_vars({test_var: "modified"}):
                assert os.environ.get(test_var) == "modified"

            assert os.environ.get(test_var) == original_value
        finally:
            os.environ.pop(test_var, None)

    def test_restores_on_exception(self) -> None:
        """Test that env vars are restored even when exception occurs."""
        test_var = "_HARK_TEST_VAR_4"
        original_value = "original"
        os.environ[test_var] = original_value

        try:
            with pytest.raises(ValueError):
                with env_vars({test_var: "modified"}):
                    assert os.environ.get(test_var) == "modified"
                    raise ValueError("test error")

            # After exception, original value should be restored
            assert os.environ.get(test_var) == original_value
        finally:
            os.environ.pop(test_var, None)

    def test_removes_new_var_on_exception(self) -> None:
        """Test that new env vars are removed even when exception occurs."""
        test_var = "_HARK_TEST_VAR_5"
        os.environ.pop(test_var, None)

        with pytest.raises(ValueError):
            with env_vars({test_var: "temporary"}):
                raise ValueError("test error")

        assert test_var not in os.environ

    def test_multiple_env_vars(self) -> None:
        """Test setting multiple env vars at once."""
        vars_to_set = {
            "_HARK_TEST_VAR_A": "value_a",
            "_HARK_TEST_VAR_B": "value_b",
            "_HARK_TEST_VAR_C": "value_c",
        }
        # Clean up first
        for var in vars_to_set:
            os.environ.pop(var, None)

        with env_vars(vars_to_set):
            for var, value in vars_to_set.items():
                assert os.environ.get(var) == value

        # All vars should be removed
        for var in vars_to_set:
            assert var not in os.environ

    def test_empty_dict(self) -> None:
        """Test with empty dictionary (no-op)."""
        # Should not raise
        with env_vars({}):
            pass

    def test_overwrites_and_restores(self) -> None:
        """Test overwriting existing var and restoring it."""
        test_var = "_HARK_TEST_VAR_6"
        os.environ[test_var] = "first"

        try:
            with env_vars({test_var: "second"}):
                assert os.environ.get(test_var) == "second"

            assert os.environ.get(test_var) == "first"
        finally:
            os.environ.pop(test_var, None)

    def test_nested_contexts(self) -> None:
        """Test nested env_vars contexts."""
        test_var = "_HARK_TEST_VAR_7"
        os.environ.pop(test_var, None)

        with env_vars({test_var: "outer"}):
            assert os.environ.get(test_var) == "outer"

            with env_vars({test_var: "inner"}):
                assert os.environ.get(test_var) == "inner"

            # After inner context, should be back to outer
            assert os.environ.get(test_var) == "outer"

        # After outer context, should be removed
        assert test_var not in os.environ

    def test_mixed_new_and_existing_vars(self) -> None:
        """Test mix of new vars and existing vars."""
        existing_var = "_HARK_TEST_VAR_8"
        new_var = "_HARK_TEST_VAR_9"

        os.environ[existing_var] = "original"
        os.environ.pop(new_var, None)

        try:
            with env_vars({existing_var: "modified", new_var: "new_value"}):
                assert os.environ.get(existing_var) == "modified"
                assert os.environ.get(new_var) == "new_value"

            assert os.environ.get(existing_var) == "original"
            assert new_var not in os.environ
        finally:
            os.environ.pop(existing_var, None)
            os.environ.pop(new_var, None)

    def test_pulse_source_scenario(self) -> None:
        """Test realistic PULSE_SOURCE scenario for audio recording."""
        pulse_source = "PULSE_SOURCE"
        original = os.environ.get(pulse_source)

        try:
            # Remove if exists
            os.environ.pop(pulse_source, None)

            with env_vars({pulse_source: "alsa_output.pci.monitor"}):
                assert os.environ.get(pulse_source) == "alsa_output.pci.monitor"

            # Should be removed after context
            assert pulse_source not in os.environ
        finally:
            # Restore original state
            if original is not None:
                os.environ[pulse_source] = original
            else:
                os.environ.pop(pulse_source, None)
