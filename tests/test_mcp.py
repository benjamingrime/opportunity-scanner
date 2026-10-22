import json

HEADERS = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}


def rpc(client, method, params=None, id=1):
    r = client.post("/mcp", headers=HEADERS, json={"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})
    assert r.status_code == 200, r.text
    return r.json()["result"]


def call(client, name, arguments):
    result = rpc(client, "tools/call", {"name": name, "arguments": arguments})
    assert not result.get("isError"), result
    return json.loads(result["content"][0]["text"])


def test_mcp_serves_the_read_only_tools(client):
    c = client
    rpc(c, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                          "clientInfo": {"name": "test", "version": "1"}})
    listed = rpc(c, "tools/list")["tools"]
    assert {t["name"] for t in listed} == {"list_roles", "grep_titles", "get_role", "compare_roles", "scan"}
    assert all(t["annotations"]["readOnlyHint"] for t in listed)

    assert call(c, "grep_titles", {"pattern": "payroll"})["by_role"][0]["soc"] == "4122"
    assert call(c, "scan", {"roles": [{"soc": "4122", "value": 3}]})["headcount"] == 3
