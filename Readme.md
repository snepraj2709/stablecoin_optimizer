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
```

### 2. Create Project Structure

Create the following directory structure:

```bash
mkdir -p agents data config api tests
touch optimizer.py
touch requirements.txt
touch .env.template
touch README.md
```

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

### 5. Copy Source Files

Copy all the provided source files into their respective directories:

**Core Files:**
- `optimizer.py` → Root directory
- `data/generate_data.py` → data/ directory

**Agent Files:**
- `agents/market_data_agent.py`
- `agents/routing_agent.py`
- `agents/compliance_agent.py`

**API Files:**
- `api/main.py`

**Demo Files:**
- `run_demo.py` → Root directory

### 6. Create Empty __init__.py Files

```bash
# Make packages importable
touch agents/__init__.py
touch config/__init__.py
touch api/__init__.py
touch tests/__init__.py
```

### 7. Generate Dummy Data

```bash
# Generate 100 sample transfers
python data/generate_data.py
```

This creates:
- `data/stablecoin_transfers.csv` (100 transfers)
- `data/sample_transfers.json` (10 sample transfers)

Expected output:
```
Generated 100 stablecoin transfer records

Summary Statistics:
Average transfer amount: $150,450.23
Average total cost (bps): 32.45
Average settlement time: 245 seconds
Settlement success rate: 92.0%
```

### 8. Test the Optimizer

#### Option A: Run Demo Script

```bash
python run_demo.py
```

This runs 5 comprehensive demos showcasing all features.

#### Option B: Run Main Optimizer

```bash
python optimizer.py
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

### 10. Test API Endpoints

#### Using curl:

```bash
# Health check
curl http://localhost:8000/health

# Optimize transfer
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "source_currency": "USD",
    "dest_currency": "USDC",
    "amount": 100000,
    "dest_chain": "Ethereum",
    "max_cost_bps": 50,
    "max_settlement_time_sec": 300,
    "region": "US",
    "kyc_verified": true
  }'
```

#### Using Python:

```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/optimize',
    json={
        'source_currency': 'USD',
        'dest_currency': 'USDC',
        'amount': 100000,
        'dest_chain': 'Ethereum',
        'max_cost_bps': 50,
        'max_settlement_time_sec': 300,
        'region': 'US',
        'kyc_verified': True
    }
)

print(response.json())
```

## Verification Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] `.env` file configured with API keys
- [ ] Dummy data generated successfully
- [ ] Optimizer runs without errors
- [ ] API server starts successfully
- [ ] API endpoints respond correctly
- [ ] Swagger docs accessible

## Troubleshooting

### Issue: Import Errors

```bash
# Make sure you're in the project root
pwd

# Verify Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: OpenAI API Errors

```python
# Test API key
python -c "import openai; print('API key valid')"

# Check environment variable
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: Port Already in Use

```bash
# Kill process on port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Module Not Found

```bash
# Ensure all __init__.py files exist
touch agents/__init__.py
touch config/__init__.py
touch api/__init__.py

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=optimizer --cov-report=html

# Run specific test file
pytest tests/test_optimizer.py -v
```

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
   python run_demo.py
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Add new feature"
   ```

### Adding New Agents

1. Create new file: `agents/new_agent.py`
2. Implement agent class with async methods
3. Import in `optimizer.py`
4. Add to orchestration workflow
5. Add tests in `tests/test_agents/`

### Code Style

```bash
# Format code
black optimizer.py agents/*.py

# Check style
flake8 optimizer.py agents/*.py

# Type checking
mypy optimizer.py
```

## Production Deployment

### Using Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build image
docker build -t stablecoin-optimizer .

# Run container
docker run -p 8000:8000 --env-file .env stablecoin-optimizer
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
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

### Add Logging

```python
import logging

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

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

## Next Steps

1. ✅ **Setup Complete** - System is operational
2. 📊 **Explore Data** - Analyze generated transfers
3. 🧪 **Run Demos** - Test all features
4. 🔧 **Customize** - Modify for your use case
5. 📈 **Monitor** - Set up dashboards
6. 🚀 **Deploy** - Move to production

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Create GitHub issue
- **Questions**: Check FAQ in README.md
- **Updates**: `git pull origin main`

## Additional Resources

- [OpenAI API Docs](https://platform.openai.com/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Pandas Docs](https://pandas.pydata.org/docs)
- [AsyncIO Docs](https://docs.python.org/3/library/asyncio.html)

---

**Setup Complete! 🎉**

Your stablecoin optimizer is ready to use. Start with `python run_demo.py` to see it in action!