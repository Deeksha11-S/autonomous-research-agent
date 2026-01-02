from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from typing import Dict, Any
import uuid
from datetime import datetime
from loguru import logger

from backend.config import settings
from backend.utils import setup_logging
from backend.agents.orchestrator import ResearchOrchestrator

# Setup logging
logger = setup_logging()

app = FastAPI(
    title="Autonomous AI Research Assistant API",
    description="Fully autonomous multi-agent research system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_research_sessions: Dict[str, Dict[str, Any]] = {}
orchestrator_pool: Dict[str, ResearchOrchestrator] = {}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Autonomous AI Research Assistant",
        "version": "1.0.0",
        "status": "operational",
        "agents": len(settings.enable_agents),
        "max_iterations": settings.max_iterations
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_research_sessions)
    }


@app.post("/research/start")
async def start_research():
    """Start a new research session"""
    session_id = str(uuid.uuid4())

    # Initialize orchestrator
    orchestrator = ResearchOrchestrator(session_id)
    orchestrator_pool[session_id] = orchestrator

    # Initialize session
    active_research_sessions[session_id] = {
        "id": session_id,
        "started_at": datetime.now().isoformat(),
        "status": "initializing",
        "progress": 0,
        "current_step": "domain_discovery",
        "messages": [],
        "results": None
    }

    logger.info(f"Started research session: {session_id}")

    return {
        "session_id": session_id,
        "message": "Research session started",
        "next_step": "connect_websocket"
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    if session_id not in active_research_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    await websocket.accept()
    session = active_research_sessions[session_id]
    orchestrator = orchestrator_pool[session_id]

    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "session_id": session_id,
            "status": session["status"],
            "progress": session["progress"]
        })

        # Start research in background
        research_task = asyncio.create_task(
            orchestrator.run_full_pipeline(
                progress_callback=lambda msg, prog: send_progress(websocket, msg, prog, session_id)
            )
        )

        # Listen for client messages (e.g., pause, cancel)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)

                if data.get("action") == "cancel":
                    research_task.cancel()
                    await websocket.send_json({
                        "type": "cancelled",
                        "message": "Research cancelled by user"
                    })
                    break

            except asyncio.TimeoutError:
                # Check if research is complete
                if research_task.done():
                    break
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

    finally:
        # Cleanup
        if session_id in orchestrator_pool:
            del orchestrator_pool[session_id]
        if session_id in active_research_sessions:
            del active_research_sessions[session_id]


async def send_progress(websocket: WebSocket, message: str, progress: float, session_id: str):
    """Send progress update via WebSocket"""
    try:
        await websocket.send_json({
            "type": "progress",
            "session_id": session_id,
            "message": message,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        })

        # Update session
        if session_id in active_research_sessions:
            session = active_research_sessions[session_id]
            session["messages"].append({
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            })
            session["progress"] = progress

    except Exception as e:
        logger.error(f"Failed to send progress: {e}")


@app.get("/research/{session_id}/status")
async def get_research_status(session_id: str):
    """Get research session status"""
    if session_id not in active_research_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return active_research_sessions[session_id]


@app.get("/research/{session_id}/results")
async def get_research_results(session_id: str):
    """Get research results"""
    if session_id not in active_research_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_research_sessions[session_id]
    if not session.get("results"):
        raise HTTPException(status_code=404, detail="Results not ready")

    return session["results"]


@app.post("/research/{session_id}/cancel")
async def cancel_research(session_id: str):
    """Cancel research session"""
    if session_id not in active_research_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel orchestrator if exists
    if session_id in orchestrator_pool:
        orchestrator_pool[session_id].cancel()
        del orchestrator_pool[session_id]

    # Update session
    session = active_research_sessions[session_id]
    session["status"] = "cancelled"
    session["ended_at"] = datetime.now().isoformat()

    return {"message": "Research cancelled", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )