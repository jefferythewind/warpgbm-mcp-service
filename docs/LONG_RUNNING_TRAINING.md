# Long-Running Training Support

## Current Status

✅ **GPU delegation working!** WarpGBM requests properly route to GPU function.

## Problem

Training can take a long time (especially for large datasets or deep models):
- LightGBM on CPU: 1-60 seconds (typical)
- WarpGBM on GPU: 5-300 seconds (depending on data size)
- Large datasets: Could take minutes

**Current behavior:** Client waits synchronously → timeout risk, poor UX

---

## Solutions

### Option 1: Async Job Queue (Recommended for production)

**Architecture:**
```
Client → POST /train → Job ID returned immediately
Client → GET /jobs/{job_id} → Poll for status
Client → GET /jobs/{job_id}/result → Get trained model when done
```

**Pros:**
- No timeout issues
- Client can poll/reconnect
- Can show progress (0%, 25%, 50%, etc.)
- Works great with MCP agents

**Cons:**
- Need job storage (Redis, Modal Dict, or database)
- More complex

**Implementation:**
```python
# Create job
@app.post("/train/async")
async def train_async(request: TrainRequest):
    job_id = str(uuid.uuid4())
    # Store job in Modal Dict or Redis
    jobs[job_id] = {"status": "pending", "created_at": time.time()}
    # Kick off training in background
    train_background.spawn(job_id, request)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}

# Check status
@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],  # pending, running, completed, failed
        "progress": job.get("progress", 0),  # 0-100
        "result_url": f"/jobs/{job_id}/result" if job["status"] == "completed" else None
    }

# Get result
@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "Result not available")
    return TrainResponse(**job["result"])
```

---

### Option 2: Server-Sent Events (SSE) Streaming

**Architecture:**
```
Client → POST /train/stream → SSE connection
Server → event: progress data: 25
Server → event: progress data: 50
Server → event: complete data: {model_artifact}
```

**Pros:**
- Real-time updates
- No polling needed
- Great UX

**Cons:**
- Client must stay connected
- Not all agents support SSE well
- Modal function timeouts still apply

**Implementation:**
```python
from fastapi.responses import StreamingResponse

@app.post("/train/stream")
async def train_stream(request: TrainRequest):
    async def event_generator():
        yield f"event: started\ndata: Training {request.model_type}\n\n"
        
        # Train with progress callbacks
        for progress in range(0, 101, 10):
            yield f"event: progress\ndata: {progress}\n\n"
            await asyncio.sleep(0.1)
        
        # Send result
        model_artifact = "..."  # Trained model
        yield f"event: complete\ndata: {json.dumps({'artifact': model_artifact})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### Option 3: Webhooks

**Architecture:**
```
Client → POST /train with callback_url
Server → Trains async
Server → POST {callback_url} with result when done
```

**Pros:**
- Client doesn't wait
- Simple for server

**Cons:**
- Client needs public URL
- Not ideal for MCP agents

---

### Option 4: Increase Timeouts (Quick fix)

**Current:** 15 minutes timeout on Modal

**Update:**
```python
@app.function(
    timeout=3600,  # 1 hour
)
```

**Pros:**
- No code changes to API
- Works for most cases

**Cons:**
- Still synchronous (bad UX)
- Client could timeout before server
- Expensive (GPU sitting idle while serializing)

---

## Recommendation

**For WarpGBM MCP Service:**

1. **Short term** (now): Increase timeout to 30 minutes
2. **Medium term** (next week): Add async job queue with `/train/async` + `/jobs/{id}`
3. **Long term** (later): Add SSE streaming for progress updates

### Why async jobs?

- **MCP agents work great with polling** - they can check status while doing other tasks
- **No connection timeouts** - job runs independently
- **Progress updates** - can show "Training 45% complete..."
- **Retry friendly** - if connection drops, agent can reconnect and check status
- **Cost efficient** - GPU shuts down immediately after training, not waiting for client

---

## Quick Win: Progress Callbacks

Even with sync training, we can add progress:

```python
# In modal_app.py GPU function
def train_warpgbm_gpu(X, y, **params):
    # Add progress callback
    def progress_callback(iteration, total):
        print(f"Progress: {iteration}/{total} ({iteration/total*100:.1f}%)")
    
    model = WarpGBMClassifier(**params)
    model.fit(X, y, callback=progress_callback)  # If WarpGBM supports it
    return serialize_model_joblib(model)
```

Agents can see progress in logs!

---

## Current Architecture Status

✅ **GPU delegation**: WarpGBM → GPU function  
✅ **Cost control**: Max 1 GPU, 60s scaledown  
✅ **LightGBM**: CPU-only, fast  
✅ **Caching**: artifact_id for fast predictions  
✅ **Error handling**: Proper 501/503 responses  

🔄 **Next**: Async job queue for long-running training

---

## Storage Options for Jobs

1. **Modal Dict** (easiest for Modal deployment)
   ```python
   jobs_dict = modal.Dict.from_name("training-jobs")
   ```

2. **Redis** (best for production)
   - Fast
   - TTL support
   - Pub/sub for live updates

3. **Database** (PostgreSQL, MySQL)
   - Persistent
   - Queryable
   - Good for job history

For MVP: **Modal Dict** is perfect!

