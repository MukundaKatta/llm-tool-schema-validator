"""Validate tool schemas against Anthropic/OpenAI calling specification.

:class:`ToolSchemaValidator` checks that each tool dict conforms to the
structure expected by major LLM providers.  It distinguishes hard errors
(will cause a provider API rejection) from warnings (likely mis-use).

Example::

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
    print(result.is_valid)   # True
    print(result.violations) # []
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Valid JSON-Schema primitive types
_VALID_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


class ViolationSeverity(str, Enum):
    """How serious a violation is."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class SchemaViolation:
    """A single validation finding.

    Attributes:
        field:    Dot-path of the offending field (e.g. ``"input_schema.type"``).
        message:  Human-readable description of the problem.
        severity: :class:`ViolationSeverity` indicating importance.
    """

    field: str
    message: str
    severity: ViolationSeverity = ViolationSeverity.ERROR

    @property
    def is_error(self) -> bool:
        """``True`` when severity is ERROR."""
        return self.severity == ViolationSeverity.ERROR

    @property
    def is_warning(self) -> bool:
        """``True`` when severity is WARNING."""
        return self.severity == ViolationSeverity.WARNING

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
        }

    def __repr__(self) -> str:
        return (
            f"SchemaViolation(field={self.field!r},"
            f" severity={self.severity.value!r})"
        )


@dataclass
class ToolSchemaValidationResult:
    """Validation outcome for a single tool definition.

    Attributes:
        tool_name:  Name of the tool (empty string if the name was absent).
        violations: List of :class:`SchemaViolation` objects found.
    """

    tool_name: str
    violations: list[SchemaViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """``True`` when there are no ERROR-severity violations."""
        return not any(v.is_error for v in self.violations)

    @property
    def errors(self) -> list[SchemaViolation]:
        """All ERROR violations."""
        return [v for v in self.violations if v.is_error]

    @property
    def warnings(self) -> list[SchemaViolation]:
        """All WARNING violations."""
        return [v for v in self.violations if v.is_warning]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "tool_name": self.tool_name,
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
        }

    def __repr__(self) -> str:
        n = len(self.violations)
        return (
            f"ToolSchemaValidationResult("
            f"tool={self.tool_name!r}, valid={self.is_valid}, violations={n})"
        )


class ToolSchemaValidator:
    """Validate tool dicts against Anthropic/OpenAI calling conventions.

    The validator accepts both Anthropic-style (``input_schema``) and
    OpenAI-style (``parameters``) tool definitions, but requires exactly
    one of the two schema keys to be present.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate_tool(self, tool: Any) -> ToolSchemaValidationResult:
        """Validate a single tool dict.

        Args:
            tool: A dict representing one tool definition.

        Returns:
            :class:`ToolSchemaValidationResult` with any findings.
        """
        violations: list[SchemaViolation] = []

        # ---- top-level type check ----
        if not isinstance(tool, dict):
            violations.append(
                SchemaViolation(
                    field="<root>",
                    message="tool must be a dict",
                )
            )
            return ToolSchemaValidationResult(tool_name="", violations=violations)

        # ---- name ----
        name = tool.get("name", "")
        if not name:
            violations.append(
                SchemaViolation(
                    field="name",
                    message="'name' is required and must be non-empty",
                )
            )
        elif not isinstance(name, str):
            violations.append(
                SchemaViolation(field="name", message="'name' must be a string")
            )
            name = str(name)

        # ---- description ----
        if "description" not in tool:
            violations.append(
                SchemaViolation(
                    field="description",
                    message=(
                        "'description' is missing"
                        " (recommended for good LLM behavior)"
                    ),
                    severity=ViolationSeverity.WARNING,
                )
            )
        elif not isinstance(tool["description"], str):
            violations.append(
                SchemaViolation(
                    field="description",
                    message="'description' must be a string",
                )
            )

        # ---- schema key ----
        has_input_schema = "input_schema" in tool
        has_parameters = "parameters" in tool

        if not has_input_schema and not has_parameters:
            violations.append(
                SchemaViolation(
                    field="input_schema",
                    message=(
                        "Neither 'input_schema' (Anthropic) nor 'parameters'"
                        " (OpenAI) is present"
                    ),
                )
            )
            return ToolSchemaValidationResult(
                tool_name=str(name), violations=violations
            )

        if has_input_schema and has_parameters:
            violations.append(
                SchemaViolation(
                    field="input_schema",
                    message=(
                        "Both 'input_schema' and 'parameters' are present;"
                        " use one or the other"
                    ),
                    severity=ViolationSeverity.WARNING,
                )
            )

        schema_key = "input_schema" if has_input_schema else "parameters"
        schema = tool[schema_key]

        self._validate_schema(schema, prefix=schema_key, violations=violations)

        return ToolSchemaValidationResult(
            tool_name=str(name), violations=violations
        )

    def validate_tools(
        self, tools: Any
    ) -> list[ToolSchemaValidationResult]:
        """Validate a list of tool dicts.

        Args:
            tools: A list of tool definition dicts.

        Returns:
            One :class:`ToolSchemaValidationResult` per tool.
        """
        if not isinstance(tools, list):
            return [
                ToolSchemaValidationResult(
                    tool_name="",
                    violations=[
                        SchemaViolation(
                            field="<root>",
                            message="tools must be a list",
                        )
                    ],
                )
            ]
        return [self.validate_tool(t) for t in tools]

    # ------------------------------------------------------------------
    # Internal schema walking
    # ------------------------------------------------------------------

    def _validate_schema(
        self,
        schema: Any,
        prefix: str,
        violations: list[SchemaViolation],
    ) -> None:
        """Recursively validate a JSON-Schema-style dict."""
        if not isinstance(schema, dict):
            violations.append(
                SchemaViolation(
                    field=prefix,
                    message=f"'{prefix}' must be a dict",
                )
            )
            return

        # ---- type ----
        if "type" not in schema:
            violations.append(
                SchemaViolation(
                    field=f"{prefix}.type",
                    message="'type' is missing (expected \"object\" at top level)",
                )
            )
        else:
            type_val = schema["type"]
            if not isinstance(type_val, str):
                violations.append(
                    SchemaViolation(
                        field=f"{prefix}.type",
                        message="'type' must be a string",
                    )
                )
            elif type_val not in _VALID_TYPES:
                violations.append(
                    SchemaViolation(
                        field=f"{prefix}.type",
                        message=(
                            f"'type' value {type_val!r} is not a valid"
                            f" JSON Schema type; expected one of {sorted(_VALID_TYPES)}"
                        ),
                    )
                )
            elif type_val != "object" and prefix in ("input_schema", "parameters"):
                violations.append(
                    SchemaViolation(
                        field=f"{prefix}.type",
                        message=(
                            f"Top-level schema 'type' must be \"object\","
                            f" got {type_val!r}"
                        ),
                    )
                )

        # ---- properties ----
        props = schema.get("properties")
        if props is not None:
            if not isinstance(props, dict):
                violations.append(
                    SchemaViolation(
                        field=f"{prefix}.properties",
                        message="'properties' must be a dict",
                    )
                )
            else:
                for prop_name, prop_schema in props.items():
                    prop_prefix = f"{prefix}.properties.{prop_name}"
                    if not isinstance(prop_schema, dict):
                        violations.append(
                            SchemaViolation(
                                field=prop_prefix,
                                message="property schema must be a dict",
                            )
                        )
                        continue
                    if "type" not in prop_schema:
                        violations.append(
                            SchemaViolation(
                                field=f"{prop_prefix}.type",
                                message="property is missing 'type'",
                                severity=ViolationSeverity.WARNING,
                            )
                        )
                    elif prop_schema["type"] not in _VALID_TYPES:
                        violations.append(
                            SchemaViolation(
                                field=f"{prop_prefix}.type",
                                message=(
                                    f"property 'type' {prop_schema['type']!r}"
                                    f" is not a valid JSON Schema type"
                                ),
                            )
                        )

        # ---- required ----
        required = schema.get("required")
        if required is not None:
            if not isinstance(required, list):
                violations.append(
                    SchemaViolation(
                        field=f"{prefix}.required",
                        message="'required' must be a list",
                    )
                )
            else:
                for item in required:
                    if not isinstance(item, str):
                        violations.append(
                            SchemaViolation(
                                field=f"{prefix}.required",
                                message=(
                                    f"'required' entries must be strings,"
                                    f" got {item!r}"
                                ),
                            )
                        )
                # Check that required items exist in properties
                if props is not None and isinstance(props, dict):
                    for req in required:
                        if isinstance(req, str) and req not in props:
                            violations.append(
                                SchemaViolation(
                                    field=f"{prefix}.required",
                                    message=(
                                        f"required field {req!r} is not listed"
                                        f" in 'properties'"
                                    ),
                                )
                            )
