"""Tool registry with automatic JSON schema generation from function signatures."""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Registered tool with metadata."""
    name: str
    func: Callable
    description: str
    parameters: dict[str, Any]
    required: list[str]
    permission: str = "safe"  # "safe", "confirm", "blocked"


class ToolRegistry:
    """Registry for tools with automatic schema generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._permissions: dict[str, str] = {}

    def set_permissions(self, permissions: dict[str, str]) -> None:
        """Set tool permissions from config."""
        self._permissions = permissions
        for name, perm in permissions.items():
            if name in self._tools:
                self._tools[name].permission = perm

    def register(self, func: Callable) -> Callable:
        """Decorator to register a tool."""
        name = func.__name__
        doc = inspect.getdoc(func) or ""
        description = doc.strip().split("\n")[0] if doc else ""

        # Get type hints
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Build JSON schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            param_schema = self._type_to_schema(param_type)
            properties[param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        tool = Tool(
            name=name,
            func=func,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            required=required,
            permission=self._permissions.get(name, "safe"),
        )

        self._tools[name] = tool
        logger.debug("Registered tool: %s (permission: %s)", name, tool.permission)
        return func

    def _type_to_schema(self, typ: Any) -> dict[str, Any]:
        """Convert Python type to JSON schema."""
        # Handle basic types
        if typ is str:
            return {"type": "string"}
        if typ is int:
            return {"type": "integer"}
        if typ is float:
            return {"type": "number"}
        if typ is bool:
            return {"type": "boolean"}

        # Handle Optional[T]
        if hasattr(typ, "__origin__"):
            origin = typ.__origin__
            args = typ.__args__

            if origin is list or origin is list:
                if args:
                    return {"type": "array", "items": self._type_to_schema(args[0])}
                return {"type": "array"}
            if origin is dict:
                return {"type": "object"}
            if origin is type(None) or (hasattr(origin, "__name__") and origin.__name__ == "Union"):
                # Optional[T] is Union[T, None]
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    schema = self._type_to_schema(non_none[0])
                    # Could add nullable: true but keeping simple
                    return schema

        # Default to string
        return {"type": "string"}

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools (for LLM function calling)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
            if tool.permission != "blocked"
        ]

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a tool by name with arguments."""
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")

        if tool.permission == "blocked":
            raise PermissionError(f"Tool {tool_name} is blocked")

        if tool.permission == "confirm":
            # In a real implementation, this would ask for confirmation
            # For now, we log and proceed (the confirm logic is in the router)
            logger.warning("Tool %s requires confirmation", tool_name)

        # Call the function with the provided kwargs
        return tool.func(**kwargs)


# Global registry instance
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def tool(func: Callable) -> Callable:
    """Decorator to register a tool."""
    return get_registry().register(func)