"""
VisionSCADA AI Twin — Model Context Protocol (MCP) Server
Exposes tools for AI Agents to:
  1. Inspect live Soft-PLC UDTs, OEE, SPC Carta p, and active ISA-18.2 alarms (`get_vision_cell_state`)
  2. Query analytics, Defect Pareto, Supplier Global Map & Model Registry (`get_mlops_and_spc_analytics`)
  3. Write RBAC-validated Soft-PLC setpoints (`write_plc_setpoint`)
  4. Inject Demo Scenarios (`inject_demo_scenario`)
  5. Query the Read-Only SQL Guardrailed Copilot (`ask_quality_copilot`)
  6. Deploy/Sync the 100% Native Ignition Perspective View into the `VisionSCADA_AI_Twin` and `DigitalTwin` Ignition projects (`deploy_ignition_perspective_view`)
"""
from __future__ import annotations
import json
from typing import Any, Dict
import httpx
from ignition_sync import deploy_to_ignition_container

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        FastMCP = None

API_BASE_URL = "http://127.0.0.1:8020/api"


def _api_get(path: str) -> Dict[str, Any]:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{API_BASE_URL}{path}")
        resp.raise_for_status()
        return resp.json()


def _api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(f"{API_BASE_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


def deploy_ignition_perspective_view(project_name: str = "VisionSCADA_AI_Twin") -> str:
    """
    Diseña e inyecta la vista 100% nativa de Ignition Perspective (`Celda_Vision_AOI_MLOps`)
    y los 4 UDTs OPC-UA directamente dentro del contenedor `twin-ignition` (tanto en el
    nuevo proyecto `VisionSCADA_AI_Twin` como en `DigitalTwin`).
    """
    res = deploy_to_ignition_container(container_name="twin-ignition", project_name=project_name)
    return json.dumps(res, indent=2, ensure_ascii=False)


if FastMCP is not None:
    mcp = FastMCP("VisionSCADA-AI-Twin-MCP")
    mcp.tool()(deploy_ignition_perspective_view)

    @mcp.tool()
    def get_vision_cell_state() -> str:
        """Obtiene el estado en vivo de los 4 UDTs del Soft-PLC, OEE, última inspección Edge y alarmas ISA-18.2."""
        return json.dumps(_api_get("/state"), indent=2, ensure_ascii=False)

    @mcp.tool()
    def get_mlops_and_spc_analytics() -> str:
        """Obtiene subgrupos SPC Carta p (Western Electric), Pareto, Mapa de Proveedores y Registro de Modelos."""
        return json.dumps(_api_get("/analytics"), indent=2, ensure_ascii=False)

    @mcp.tool()
    def inject_demo_scenario(scenario: str = "bad_supplier_lot", active: bool = True) -> str:
        """Activa/desactiva escenarios: 'bad_supplier_lot', 'dirty_lens_drift', 'cloud_outage', 'reset_all'."""
        return json.dumps(
            _api_post("/scenario", {"scenario": scenario, "active": active, "user_name": "mcp_agent", "user_role": "Plant_Manager"}),
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def ask_quality_copilot(question: str) -> str:
        """Consulta al Copiloto IA con Guardrails SQL Read-Only (`copilot_ro`) y citas verificables."""
        return json.dumps(_api_post("/copilot/ask", {"question": question}), indent=2, ensure_ascii=False)


if __name__ == "__main__" and FastMCP is not None:
    mcp.run()
