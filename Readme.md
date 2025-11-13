# Complete Setup Guide

This guide will walk you through setting up the Stablecoin Routing Optimization Engine from scratch.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- OpenAI API key (required)
- Google API key (optional)
- 2GB free disk space

## Step-by-Step Setup

### 1. Clone or Create Project Directory

```bash
# Create project directory
mkdir stablecoin-optimizer
cd stablecoin-optimizer

### 3. Install Python Dependencies

Create `requirements.txt` with the provided content, then:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy template
cp .env.template .env

# Edit .env file
nano .env  # or use your preferred editor
```

Add your API keys to `.env`:

```bash
# REQUIRED
OPENAI_API_KEY=sk-your-actual-openai-key-here

# OPTIONAL (for enhanced features)
GOOGLE_API_KEY=your-google-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Application settings
LOG_LEVEL=INFO
CACHE_TTL_SECONDS=30
```

**Getting API Keys:**

- **OpenAI**: https://platform.openai.com/api-keys
- **Google**: https://makersuite.google.com/app/apikey
- **Anthropic**: https://console.anthropic.com/

### 7. Generate Dummy Data

```bash
# Generate 100 sample transfers
python main.py
```

This creates:
- `config/generated_transfers.csv` (100 dummy transfers)
- `config/normalized_transactions.csv` (100 normalized transfers)
- `config/optimization_results.json` (100 optimized transfers)

Expected output:
```
Generated 100 stablecoin transfer records

Summary Statistics:
Average transfer amount: $150,450.23
Average total cost (bps): 32.45
Average settlement time: 245 seconds
Settlement success rate: 92.0%
```



### Data flow architecture
flowchart TD
  subgraph INGEST["1. RAW INPUTS"]
    A1[Stablecoin API] --> N
    A2[CSV / Webhook / ERP / Synthetic] --> N
  end

  subgraph NORM["2. NORMALIZER"]
    N[Normalizer: Transaction dataclass / clean fields / types]
  end

  subgraph ADAPT["3. ADAPTER"]
    AD[Adapter: enrich with candidate routes / weights / pre-checks]
  end

  subgraph OPT["4. OPTIMIZER"]
    O[Optimizer: MIP solver (α·cost + β·time + γ·risk)]
    O[Hard constraints: (max cost, max time, max slippage)]
    O[Context flags (cross-border, high-value, fast settlement)]
  end

  subgraph ANALYTICS["5. TREASURY ANALYTICS"]
    AN[KPI timeseries / idle capital / FX exposure]
  end

  subgraph AI["6. AI INSIGHTS"]
    AIo[Summaries / anomalies / recommendations]
  end

  subgraph DASH["7. DASHBOARD"]
    D[Streamlit / React UI]
  end

  %% flow
  N --> AD
  AD --> O
  O --> AN
  AN --> AIo
  AIo --> D


### 8. Test the Optimizer

#### Option A: Run Demo Script

```bash
python main.py
streamlit run dashboard/dashboard.py
```

Expected output:
```
[OPTIMIZER] Initialized StablecoinOptimizer

[OPTIMIZER] Processing transfer: TEST_001
  Route: USD -> USDC
  Amount: 100,000.00

[STEP 1] Compliance verification...
[STEP 2] Fetching market data...
[STEP 3] Assessing liquidity...
[STEP 4] Generating optimal routes...
[STEP 5] Evaluating route risks...
[STEP 6] Selecting optimal route...

[OPTIMIZER] Optimization complete in 245.32ms
  Selected route cost: 28.45 bps
  Estimated settlement: 180s
```

### 9. Start the API Server

```bash
# Start FastAPI server
uvicorn api.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


## Verification Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] `.env` file configured with API keys
- [ ] Dummy data generated successfully
- [ ] Optimizer runs without errors
- [ ] API server starts successfully
- [ ] API endpoints respond correctly
- [ ] Swagger docs accessible

## Development Workflow

### Making Changes

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make changes**
   - Edit source files
   - Add tests

3. **Test changes**
   ```bash
   pytest
   python main.py
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Add new feature"
   ```

## Performance Optimization

### Enable Caching

```python
# In optimizer.py
import redis

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=6379,
    decode_responses=True
)
```

### Parallel Processing

```python
# Optimize multiple transfers in parallel
import asyncio

results = await asyncio.gather(*[
    optimizer.optimize_transfer(req)
    for req in requests
])
```

### Database Integration

```python
# For persistence
from sqlalchemy import create_engine

engine = create_engine(os.getenv('DATABASE_URL'))
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

optimization_requests = Counter(
    'optimizer_requests_total',
    'Total optimization requests'
)

optimization_duration = Histogram(
    'optimizer_duration_seconds',
    'Optimization duration'
)
```