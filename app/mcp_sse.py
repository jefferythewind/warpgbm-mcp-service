"""
MCP Server-Sent Events (SSE) endpoint for Cursor integration
Simplified version - uses REST endpoints internally
"""

import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


async def event_stream() -> AsyncGenerator[str, None]:
    """
    SSE endpoint that keeps connection alive and provides server info
    """
    # Send immediate response to avoid buffering
    yield ":"  # Empty comment line to force connection
    yield "\n"
    
    # Send initial connection event with available tools
    manifest = {
        "name": "warpgbm-mcp",
        "version": "1.0.0",
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {
                "list_models": {
                    "description": "List all available ML model backends (warpgbm, lightgbm)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                "train": {
                    "description": "Train a gradient boosting model and return portable artifacts",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "X": {"type": "array", "description": "Feature matrix (2D array)"},
                            "y": {"type": "array", "description": "Target labels"},
                            "model_type": {"type": "string", "enum": ["warpgbm", "lightgbm"], "default": "warpgbm"},
                            "objective": {"type": "string", "enum": ["regression", "binary", "multiclass"], "default": "multiclass"},
                            "n_estimators": {"type": "integer", "default": 100},
                            "learning_rate": {"type": "number", "default": 0.1},
                            "max_depth": {"type": "integer", "default": 6}
                        },
                        "required": ["X", "y"]
                    }
                },
                "predict_from_artifact": {
                    "description": "Run inference using a trained model artifact",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "X": {"type": "array", "description": "Feature matrix for prediction"},
                            "model_artifact_joblib": {"type": "string", "description": "Base64-encoded joblib model"},
                            "model_artifact_onnx": {"type": "string", "description": "Base64-encoded ONNX model (optional)"}
                        },
                        "required": ["X"]
                    }
                }
            }
        }
    }
    
    yield f"event: hello\n"
    yield f"data: {json.dumps(manifest)}\n"
    yield "\n"
    
    # Keep connection alive with periodic heartbeats
    try:
        while True:
            await asyncio.sleep(15)
            yield ": ping\n"
            yield "\n"
    except asyncio.CancelledError:
        pass


@router.get("/mcp/sse")
async def mcp_sse_get(request: Request):
    """
    MCP SSE endpoint (GET) - establishes connection and keeps it alive
    """
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "text/event-stream; charset=utf-8",
            "Transfer-Encoding": "chunked",
        }
    )


@router.post("/mcp/sse")
async def mcp_sse_post(request: Request):
    """
    MCP SSE endpoint (POST) - handles JSON-RPC tool calls
    Implements the MCP protocol for tool invocations
    """
    try:
        data = await request.json()
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")
        
        # Handle initialize method
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": "warpgbm-mcp",
                        "version": "1.0.0"
                    }
                }
            }
        
        # Handle tools/list method
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "list_models",
                            "description": "List all available ML model backends (warpgbm, lightgbm)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "train",
                            "description": "Train a gradient boosting model and return portable artifacts (joblib and/or ONNX)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "X": {"type": "array", "description": "Feature matrix (2D array of floats)"},
                                    "y": {"type": "array", "description": "Target labels (floats for regression, integers for classification)"},
                                    "model_type": {"type": "string", "enum": ["warpgbm", "lightgbm"], "default": "warpgbm", "description": "Model backend to use"},
                                    "objective": {"type": "string", "enum": ["regression", "binary", "multiclass"], "default": "multiclass", "description": "Training objective"},
                                    "n_estimators": {"type": "integer", "default": 100, "description": "Number of trees"},
                                    "learning_rate": {"type": "number", "default": 0.1, "description": "Learning rate"},
                                    "max_depth": {"type": "integer", "default": 6, "description": "Maximum tree depth"}
                                },
                                "required": ["X", "y"]
                            }
                        },
                        {
                            "name": "predict_from_artifact",
                            "description": "Run inference using a trained model artifact or artifact_id. Use artifact_id for fast predictions right after training (valid for 5 minutes).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "X": {"type": "array", "description": "Feature matrix for prediction (2D array)"},
                                    "artifact_id": {"type": "string", "description": "Temporary artifact ID from training response (fast, valid for 5 minutes)"},
                                    "model_artifact_joblib": {"type": "string", "description": "Base64-encoded, gzip-compressed joblib model"}
                                },
                                "required": ["X"]
                            }
                        },
                        {
                            "name": "get_agent_guide",
                            "description": "Get the comprehensive agent guide with examples, best practices, and troubleshooting tips for using this service",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "upload_data",
                            "description": "Upload CSV or Parquet files for training. Parses files and returns structured X and y arrays ready for training.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "file_content": {"type": "string", "description": "Base64-encoded file content"},
                                    "file_format": {"type": "string", "enum": ["csv", "parquet"], "description": "File format"},
                                    "target_column": {"type": "string", "description": "Column name for target variable (y)"},
                                    "feature_columns": {"type": "array", "description": "Column names for features (X). If not specified, all columns except target are used."}
                                },
                                "required": ["file_content", "file_format"]
                            }
                        },
                        {
                            "name": "submit_feedback",
                            "description": "Submit feedback about the service. Agents can report bugs, request features, or provide general feedback.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "feedback_type": {"type": "string", "enum": ["bug", "feature_request", "documentation", "performance", "general"], "description": "Type of feedback"},
                                    "message": {"type": "string", "description": "Feedback message"},
                                    "endpoint": {"type": "string", "description": "Related endpoint (if applicable)"},
                                    "model_type": {"type": "string", "description": "Related model type (if applicable)"},
                                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], "default": "medium", "description": "Severity level"},
                                    "agent_info": {"type": "object", "description": "Agent metadata (name, version, etc.)"}
                                },
                                "required": ["feedback_type", "message"]
                            }
                        }
                    ]
                }
            }
        
        # Handle tools/call method
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Import httpx for internal API calls
            import httpx
            
            # Handle get_agent_guide specially (reads from file)
            if tool_name == "get_agent_guide":
                try:
                    import os
                    guide_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "AGENT_GUIDE.md")
                    with open(guide_path, "r") as f:
                        guide_content = f.read()
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": guide_content
                                }
                            ]
                        }
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Failed to read agent guide: {str(e)}"
                        }
                    }
            
            # Map MCP tool names to REST endpoints
            endpoint_map = {
                "list_models": "/models",
                "train": "/train",
                "predict_from_artifact": "/predict_from_artifact",
                "upload_data": "/upload_data",
                "submit_feedback": "/feedback"
            }
            
            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
            
            # Call the internal REST API
            try:
                base_url = str(request.base_url).rstrip('/')
                full_url = f"{base_url}{endpoint}"
                
                # Transform arguments for predict_from_artifact
                request_body = arguments.copy() if arguments else {}
                if tool_name == "predict_from_artifact":
                    # MCP tool uses model_artifact_joblib, REST API uses model_artifact + format
                    # artifact_id passes through as-is
                    if "model_artifact_joblib" in request_body:
                        request_body["model_artifact"] = request_body.pop("model_artifact_joblib")
                        request_body["format"] = "joblib"
                    # artifact_id is already in request_body if provided, no transformation needed
                
                async with httpx.AsyncClient(timeout=300.0) as client:
                    if tool_name == "list_models":
                        response = await client.get(full_url)
                    else:
                        response = await client.post(full_url, json=request_body)
                
                if response.status_code == 200:
                    result_data = response.json()
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result_data, indent=2)
                                }
                            ]
                        }
                    }
                else:
                    error_detail = response.json().get("detail", response.text) if response.text else "Unknown error"
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": response.status_code,
                            "message": f"Tool execution failed: {error_detail}"
                        }
                    }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error calling tool: {str(e)}"
                    }
                }
        
        # Unknown method
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": data.get("id") if "data" in locals() else None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


@router.options("/mcp/sse")
async def mcp_sse_options():
    """Handle CORS preflight"""
    return {
        "Allow": "GET, POST, OPTIONS",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    }
