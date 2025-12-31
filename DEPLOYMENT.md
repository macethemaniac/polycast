# Deployment Guide

This guide covers deploying the Polycast arbitrage bot to various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Deployment](#local-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Systemd Service (Linux)](#systemd-service-linux)
7. [Monitoring and Logs](#monitoring-and-logs)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/polycast.git
cd polycast
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Or install as a package (recommended)
pip install -e .
```

### 4. Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
TELEGRAM_BOT_TOKEN=your_actual_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## Local Deployment

### Running the Bot

**Option 1: Using the startup script (Windows)**
```powershell
.\start_bot.bat
```

**Option 2: Direct Python execution**
```bash
# From project root
python src/polycast/bot.py
```

**Option 3: Using the installed package**
```bash
polycast-bot
```

### Running the Console Scanner

```bash
python src/polycast/main.py
# or
polycast-scan
```

## Cloud Deployment

### Deploy to Heroku

1. **Create a Heroku app:**
```bash
heroku create your-app-name
```

2. **Set environment variables:**
```bash
heroku config:set TELEGRAM_BOT_TOKEN=your_token_here
heroku config:set TELEGRAM_CHAT_ID=your_chat_id_here
```

3. **Create a Procfile:**
```
worker: python src/polycast/bot.py
```

4. **Deploy:**
```bash
git push heroku main
```

5. **Scale the worker:**
```bash
heroku ps:scale worker=1
```

### Deploy to Render (Recommended)

Render is a modern cloud platform with a generous free tier, perfect for hosting Telegram bots.

#### Method 1: Using Blueprint (Easiest)

1. **Push your code to GitHub:**
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

2. **Connect to Render:**
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`

3. **Set environment variables:**
   - In the Render dashboard, go to your service
   - Navigate to "Environment" tab
   - Add your secrets:
     - `TELEGRAM_BOT_TOKEN`: Your bot token
     - `TELEGRAM_CHAT_ID`: Your chat ID
     - `KALSHI_API_KEY`: (if using Kalshi)
     - `KALSHI_API_SECRET`: (if using Kalshi)

4. **Deploy:**
   - Click "Apply" to create the service
   - Render will automatically build and deploy
   - Your bot will start running in ~2 minutes

#### Method 2: Manual Setup

1. **Create a new Web Service:**
   - Go to [render.com](https://render.com) dashboard
   - Click "New +" → "Background Worker"
   - Connect your GitHub repository

2. **Configure the service:**
   - **Name:** `polycast-bot`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python src/polycast/bot.py`
   - **Plan:** Free (or paid for more resources)

3. **Set environment variables:**
   Add these in the "Environment" section:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   KALSHI_API_KEY=your_kalshi_key (optional)
   KALSHI_API_SECRET=your_kalshi_secret (optional)
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   PYTHONPATH=/opt/render/project/src
   ```

4. **Advanced settings (optional):**
   - **Auto-Deploy:** Enable for automatic deployments on git push
   - **Health Check Path:** Leave empty for background workers
   - **Region:** Choose closest to your location

5. **Create the service:**
   - Click "Create Background Worker"
   - Render will build and deploy automatically

#### Monitoring on Render

**View logs:**
- Go to your service dashboard
- Click on "Logs" tab
- Real-time logs will appear

**Check service status:**
- Dashboard shows: Running, Building, or Failed
- Green dot = healthy and running

**Restart service:**
- Click "Manual Deploy" → "Deploy latest commit"
- Or click "Restart" for a quick restart

#### Free Tier Limitations

Render's free tier includes:
- 750 hours/month of runtime (enough for 24/7 operation)
- 512 MB RAM
- Shared CPU
- Automatic sleep after 15 minutes of inactivity (background workers don't sleep)

**Note:** Background workers on the free tier do NOT sleep, unlike web services.

#### Troubleshooting Render Deployment

**Bot not starting:**
1. Check logs in Render dashboard
2. Verify all environment variables are set
3. Ensure `PYTHONPATH=/opt/render/project/src` is set

**Import errors:**
- Make sure `PYTHONPATH` environment variable is set
- Check that `requirements.txt` has all dependencies

**Deployment fails:**
- Check build logs for specific errors
- Verify Python version compatibility
- Ensure all files are committed to git

**Bot stops responding:**
- Check if service is running in dashboard
- Review recent logs for errors
- Verify Telegram token is valid

#### Updating Your Bot on Render

**Automatic updates (if enabled):**
```bash
git add .
git commit -m "Update bot"
git push origin main
# Render automatically deploys
```

**Manual deploy:**
- Go to Render dashboard
- Click "Manual Deploy" → "Deploy latest commit"

### Deploy to AWS EC2

1. **Launch an EC2 instance** (Ubuntu 20.04 or later recommended)

2. **SSH into your instance:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
```

4. **Clone and setup:**
```bash
git clone https://github.com/yourusername/polycast.git
cd polycast
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Configure environment variables:**
```bash
cp .env.example .env
nano .env  # Edit with your credentials
```

6. **Run as a background process** (see Systemd Service section below)

### Deploy to DigitalOcean

Similar to AWS EC2 - create a Droplet and follow the same steps.

## Docker Deployment

### Create Dockerfile

Create `Dockerfile` in project root:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy dependency files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/app/src

# Run the bot
CMD ["python", "src/polycast/bot.py"]
```

### Build and Run

```bash
# Build the image
docker build -t polycast-bot .

# Run the container
docker run -d \
  --name polycast \
  --env-file .env \
  --restart unless-stopped \
  polycast-bot
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  bot:
    build: .
    container_name: polycast-bot
    env_file: .env
    restart: unless-stopped
    volumes:
      - ./src/data:/app/src/data
```

Run with:
```bash
docker-compose up -d
```

## Systemd Service (Linux)

Create a systemd service for automatic startup and restart.

### 1. Create Service File

Create `/etc/systemd/system/polycast.service`:

```ini
[Unit]
Description=Polycast Arbitrage Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/polycast
Environment="PATH=/home/ubuntu/polycast/venv/bin"
EnvironmentFile=/home/ubuntu/polycast/.env
ExecStart=/home/ubuntu/polycast/venv/bin/python src/polycast/bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable polycast

# Start service
sudo systemctl start polycast

# Check status
sudo systemctl status polycast
```

### 3. Service Management

```bash
# Stop service
sudo systemctl stop polycast

# Restart service
sudo systemctl restart polycast

# View logs
sudo journalctl -u polycast -f
```

## Monitoring and Logs

### Application Logs

The bot uses Python's logging module. Logs are printed to stdout/stderr.

**View logs when running as systemd service:**
```bash
sudo journalctl -u polycast -f
```

**View logs in Docker:**
```bash
docker logs -f polycast
```

### Health Checks

Create a simple health check script `scripts/health_check.py`:

```python
import sys
import requests

def check_bot_health():
    try:
        # Add your health check logic here
        # For example, check if bot responds to a test command
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if check_bot_health() else 1)
```

## Troubleshooting

### Bot Not Responding

1. **Check if the bot is running:**
```bash
# Systemd
sudo systemctl status polycast

# Docker
docker ps | grep polycast
```

2. **Check logs for errors:**
```bash
# Systemd
sudo journalctl -u polycast -n 100

# Docker
docker logs polycast --tail 100
```

3. **Verify environment variables:**
```bash
# Check if TELEGRAM_BOT_TOKEN is set
echo $TELEGRAM_BOT_TOKEN
```

### Import Errors

If you get import errors, ensure Python path is set correctly:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/polycast/src"
```

Or install the package:
```bash
pip install -e .
```

### Connection Issues

1. **Check internet connectivity**
2. **Verify Telegram API is accessible**
3. **Check firewall settings** (ensure outbound HTTPS is allowed)

### Rate Limiting

If you encounter rate limiting errors:
- Reduce polling frequency
- Implement exponential backoff
- Consider using webhooks instead of polling

## Security Best Practices

1. **Never commit `.env` file** (already in `.gitignore`)
2. **Use environment variables** for all secrets
3. **Keep dependencies updated:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```
4. **Run with minimal privileges** (non-root user)
5. **Enable firewall:**
   ```bash
   sudo ufw allow 22/tcp
   sudo ufw enable
   ```

## Scaling Considerations

For high-traffic deployments:

1. **Use webhooks instead of polling** (more efficient)
2. **Implement queue system** (Redis + Celery) for background tasks
3. **Use database** (PostgreSQL/MongoDB) for persistent storage
4. **Load balancing** for multiple bot instances
5. **Caching** for API responses (Redis)

## Updating the Bot

```bash
# Stop the service
sudo systemctl stop polycast

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl start polycast
```

## Backup and Recovery

### Backup Important Files

```bash
# Backup configuration
cp .env .env.backup

# Backup data (if any)
cp -r src/data src/data.backup
```

### Automated Backups

Create a cron job for daily backups:
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /home/ubuntu/polycast/scripts/backup.sh
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/polycast/issues
- Documentation: See `README.md`, `BOT_SETUP.md`, `POLYMARKET_API.md`
