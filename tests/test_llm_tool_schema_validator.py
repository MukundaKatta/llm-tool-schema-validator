"""Tests for llm-tool-schema-validator."""

from __future__ import annotations

from llm_tool_schema_validator import (
    SchemaViolation,
    ToolSchemaValidationResult,
    ToolSchemaValidator,
    ViolationSeverity,
)

# ---------------------------------------------------------------------------
# ViolationSeverity
# ---------------------------------------------------------------------------


def test_severity_values():
    assert ViolationSeverity.ERROR.value == "error"
    assert ViolationSeverity.WARNING.value == "warning"


def test_severity_is_str():
    assert isinstance(ViolationSeverity.ERROR, str)


# ---------------------------------------------------------------------------
# SchemaViolation
# ---------------------------------------------------------------------------


def test_violation_defaults_to_error():
    v = SchemaViolation(field="name", message="missing")
    assert v.is_error
    assert not v.is_warning


def test_violation_warning():
    v = SchemaViolation(
        field="desc", message="missing", severity=ViolationSeverity.WARNING
    )
    assert v.is_warning
    assert not v.is_error


def test_violation_to_dict():
    v = SchemaViolation(field="x", message="bad", severity=ViolationSeverity.ERROR)
    d = v.to_dict()
    assert d["field"] == "x"
    assert d["message"] == "bad"
    assert d["severity"] == "error"


def test_violation_repr():
    v = SchemaViolation(field="name", message="x")
    r = repr(v)
    assert "name" in r


# ---------------------------------------------------------------------------
# ToolSchemaValidationResult
# ---------------------------------------------------------------------------


def test_result_is_valid_no_violations():
    r = ToolSchemaValidationResult(tool_name="t")
    assert r.is_valid


def test_result_invalid_with_error():
    r = ToolSchemaValidationResult(
        tool_name="t",
        violations=[SchemaViolation(field="x", message="bad")],
    )
    assert not r.is_valid


def test_result_valid_with_only_warnings():
    r = ToolSchemaValidationResult(
        tool_name="t",
        violations=[
            SchemaViolation(
                field="desc", message="missing", severity=ViolationSeverity.WARNING
            )
        ],
    )
    assert r.is_valid


def test_result_errors_and_warnings():
    r = ToolSchemaValidationResult(
        tool_name="t",
        violations=[
            SchemaViolation(field="a", message="e"),
            SchemaViolation(field="b", message="w", severity=ViolationSeverity.WARNING),
        ],
    )
    assert len(r.errors) == 1
    assert len(r.warnings) == 1


def test_result_to_dict():
    r = ToolSchemaValidationResult(tool_name="search")
    d = r.to_dict()
    assert d["tool_name"] == "search"
    assert d["is_valid"] is True
    assert d["violations"] == []


def test_result_repr():
    r = ToolSchemaValidationResult(tool_name="foo")
    assert "foo" in repr(r)


# ---------------------------------------------------------------------------
# ToolSchemaValidator — valid tool (Anthropic style)
# ---------------------------------------------------------------------------


def _good_tool() -> dict:
    return {
        "name": "search",
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
            },
            "required": ["query"],
        },
    }


def test_valid_anthropic_tool():
    v = ToolSchemaValidator()
    r = v.validate_tool(_good_tool())
    assert r.is_valid
    assert r.tool_name == "search"
    assert r.violations == []


def test_valid_openai_tool():
    tool = {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    }
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert r.is_valid


def test_valid_tool_no_properties():
    tool = {
        "name": "ping",
        "description": "Ping the server",
        "input_schema": {"type": "object"},
    }
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert r.is_valid


# ---------------------------------------------------------------------------
# ToolSchemaValidator — name errors
# ---------------------------------------------------------------------------


def test_missing_name():
    tool = {**_good_tool()}
    del tool["name"]
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid
    fields = [vl.field for vl in r.errors]
    assert "name" in fields


def test_empty_name():
    tool = {**_good_tool(), "name": ""}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_non_string_name():
    tool = {**_good_tool(), "name": 42}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


# ---------------------------------------------------------------------------
# ToolSchemaValidator — description warnings
# ---------------------------------------------------------------------------


def test_missing_description_warning():
    tool = {**_good_tool()}
    del tool["description"]
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    # is_valid because it's only a warning
    assert r.is_valid
    assert any(vl.field == "description" for vl in r.warnings)


def test_non_string_description_error():
    tool = {**_good_tool(), "description": 123}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


# ---------------------------------------------------------------------------
# ToolSchemaValidator — schema key errors
# ---------------------------------------------------------------------------


def test_missing_schema_key():
    tool = {"name": "t", "description": "d"}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_both_schema_keys_warning():
    tool = {
        "name": "t",
        "description": "d",
        "input_schema": {"type": "object"},
        "parameters": {"type": "object"},
    }
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    # may be valid (only warning), but should have the warning
    schema_fields = {"input_schema", "parameters"}
    assert any(vl.field in schema_fields for vl in r.warnings)


# ---------------------------------------------------------------------------
# ToolSchemaValidator — schema type checks
# ---------------------------------------------------------------------------


def test_schema_missing_type():
    tool = {**_good_tool()}
    tool["input_schema"] = {"properties": {}}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_schema_type_not_object():
    tool = {**_good_tool()}
    tool["input_schema"] = {"type": "string"}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_schema_invalid_type_value():
    tool = {**_good_tool()}
    tool["input_schema"] = {"type": "foobar"}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_schema_not_a_dict():
    tool = {**_good_tool()}
    tool["input_schema"] = "not a dict"
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


# ---------------------------------------------------------------------------
# ToolSchemaValidator — properties checks
# ---------------------------------------------------------------------------


def test_properties_not_dict():
    tool = {**_good_tool()}
    tool["input_schema"]["properties"] = ["not", "a", "dict"]
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_property_schema_not_dict():
    tool = {**_good_tool()}
    tool["input_schema"]["properties"]["query"] = "bad"
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_property_missing_type_warning():
    tool = {**_good_tool()}
    tool["input_schema"]["properties"]["query"] = {"description": "no type"}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert r.is_valid  # warning only
    assert any("type" in vl.field for vl in r.warnings)


def test_property_invalid_type_value():
    tool = {**_good_tool()}
    tool["input_schema"]["properties"]["query"] = {"type": "invalid_type"}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_property_type_non_string_list():
    # A non-string 'type' (e.g. a JSON Schema union list) must be reported as a
    # violation, not raise TypeError when checked against the valid-types set.
    tool = {**_good_tool()}
    tool["input_schema"]["properties"]["query"] = {"type": ["string", "null"]}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid
    assert any(vl.field.endswith(".type") for vl in r.errors)


def test_property_type_non_string_dict():
    tool = {**_good_tool()}
    tool["input_schema"]["properties"]["query"] = {"type": {"unexpected": "dict"}}
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


# ---------------------------------------------------------------------------
# ToolSchemaValidator — required checks
# ---------------------------------------------------------------------------


def test_required_not_list():
    tool = {**_good_tool()}
    tool["input_schema"]["required"] = "query"
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_required_entry_not_string():
    tool = {**_good_tool()}
    tool["input_schema"]["required"] = [42]
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid


def test_required_missing_from_properties():
    tool = {**_good_tool()}
    tool["input_schema"]["required"] = ["query", "nonexistent"]
    v = ToolSchemaValidator()
    r = v.validate_tool(tool)
    assert not r.is_valid
    assert any("nonexistent" in vl.message for vl in r.errors)


# ---------------------------------------------------------------------------
# ToolSchemaValidator — non-dict tool
# ---------------------------------------------------------------------------


def test_tool_not_dict():
    v = ToolSchemaValidator()
    r = v.validate_tool("not a dict")
    assert not r.is_valid


def test_tool_none():
    v = ToolSchemaValidator()
    r = v.validate_tool(None)
    assert not r.is_valid


# ---------------------------------------------------------------------------
# ToolSchemaValidator — validate_tools
# ---------------------------------------------------------------------------


def test_validate_tools_all_valid():
    v = ToolSchemaValidator()
    results = v.validate_tools([_good_tool(), _good_tool()])
    assert all(r.is_valid for r in results)
    assert len(results) == 2


def test_validate_tools_some_invalid():
    bad = {"name": "", "input_schema": {"type": "object"}}
    v = ToolSchemaValidator()
    results = v.validate_tools([_good_tool(), bad])
    assert results[0].is_valid
    assert not results[1].is_valid


def test_validate_tools_not_a_list():
    v = ToolSchemaValidator()
    results = v.validate_tools("not a list")
    assert len(results) == 1
    assert not results[0].is_valid


def test_validate_tools_empty_list():
    v = ToolSchemaValidator()
    results = v.validate_tools([])
    assert results == []
