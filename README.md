# llm-tool-schema-validator

Validate tool schemas against the Anthropic/OpenAI calling specification before sending. Zero dependencies.

## Install

```bash
pip install llm-tool-schema-validator
```

## Quick start

```python
from llm_tool_schema_validator import ToolSchemaValidator

validator = ToolSchemaValidator()

tool = {
    "name": "search",
    "description": "Search the web",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
}

result = validator.validate_tool(tool)
print(result.is_valid)    # True
print(result.violations)  # []

# Validate all tools at once
results = validator.validate_tools([tool1, tool2, tool3])
invalid = [r for r in results if not r.is_valid]
```

## What it checks

| Check | Severity |
|---|---|
| `name` present and non-empty string | error |
| `description` present | warning |
| `input_schema` (Anthropic) or `parameters` (OpenAI) present | error |
| Both `input_schema` and `parameters` present | warning |
| Schema is a dict | error |
| Top-level schema `type` == `"object"` | error |
| `type` is a valid JSON Schema type | error |
| `properties` is a dict | error |
| Each property schema is a dict | error |
| Each property has a `type` | warning |
| Each property `type` is a valid JSON Schema type | error |
| `required` is a list of strings | error |
| Required fields exist in `properties` | error |

## API

### `ToolSchemaValidator`

| Method | Description |
|---|---|
| `validate_tool(tool)` | Validate one tool dict → `ToolSchemaValidationResult` |
| `validate_tools(tools)` | Validate a list → `list[ToolSchemaValidationResult]` |

### `ToolSchemaValidationResult`

| Attribute | Description |
|---|---|
| `tool_name` | Name of the tool |
| `is_valid` | `True` when no ERROR violations |
| `violations` | All `SchemaViolation` objects |
| `errors` | ERROR-severity violations |
| `warnings` | WARNING-severity violations |

## License

MIT
