"""MCP server: the scanner's read-only tools for any MCP client (Claude, ChatGPT, Cursor...).

The web app serves it over Streamable HTTP at /mcp. Locally, over stdio:

  uv run python -m scanner.mcp_server
"""
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from scanner import tools

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

server = MCPServer(
    "opportunity-scanner",
    instructions=(
        "Where could AI help in a UK organisation? Roles are UK SOC 2020 unit groups (four-digit codes). "
        "Each role's working time splits into E1 'buy and train' (an off-the-shelf AI assistant could halve "
        "the time), E2 'build' (needs software or an agent connected to the organisation's systems) and E0 "
        "'people stay central'. Shares are of working time spent on tasks, from task-level exposure labels "
        "(Eloundou et al. 2023 rubric) weighted by estimated time per task (method of Hatgis-Kessell et al. "
        "2026). They are not time saved, and say nothing about jobs lost. Find a role's code with "
        "grep_titles; index titles are often inverted ('Officer, personnel')."),
)


class RoleCount(BaseModel):
    soc: str = Field(pattern=r"^\d{4}$", description="SOC 2020 unit group")
    value: float = Field(ge=0, description="Number of people")


@server.tool(annotations=READ_ONLY)
def list_roles() -> list[tuple[str, str]]:
    """All 412 UK SOC 2020 unit groups as (code, title)."""
    return tools.list_roles()


@server.tool(annotations=READ_ONLY)
def grep_titles(pattern: Annotated[str, Field(max_length=200)]) -> dict:
    """Search the ~30,000 ONS job titles (case-insensitive regular expression, e.g. 'payroll|wages').
    Returns up to 50 titles with their unit groups, and the unit groups with most matches."""
    return tools.grep_titles(pattern)


@server.tool(annotations=READ_ONLY)
def get_role(soc: Annotated[str, Field(pattern=r"^\d{4}$")]) -> dict:
    """A unit group's split of working time, its O*NET occupations and its longest tasks under each heading."""
    return tools.get_role(soc)


@server.tool(annotations=READ_ONLY)
def compare_roles(socs: Annotated[list[str], Field(max_length=20)]) -> list[dict]:
    """Exposure split and beta, side by side, for up to 20 unit groups."""
    return tools.compare_roles(socs)


@server.tool(annotations=READ_ONLY)
def scan(roles: Annotated[list[RoleCount], Field(max_length=500)]) -> dict:
    """A workforce's split of working time into the three headings, weighted by headcount, with hours a
    week for full-time staff, and the top roles for buy-and-train and for build."""
    return tools.scan([r.model_dump() for r in roles], mode="count")


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
