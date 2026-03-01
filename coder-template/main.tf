terraform {
  required_providers {
    coder = {
      source  = "coder/coder"
      version = "~> 2.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "coder" {}

provider "docker" {
  host = "npipe:////.//pipe//docker_engine"
}

data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}

# ── CODER AGENT ───────────────────────────────────────────────────────────────
resource "coder_agent" "main" {
  os   = "linux"
  arch = "amd64"

  startup_script = <<-EOT
    #!/bin/bash
    echo "=== AushadhiNet DDI Workspace Starting ==="
    cd /workspaces/ddi

    echo "Installing dependencies..."
    pip install --upgrade pip --quiet
    pip install -r requirements_docker.txt --quiet

    echo "Launching Streamlit DDI Inference App..."
    nohup streamlit run inference_app.py \
      --server.port=8501 \
      --server.address=0.0.0.0 \
      --server.headless=true \
      > /tmp/streamlit.log 2>&1 &

    echo "Done. App running on port 8501."
  EOT
}

# ── CODER APP — clickable button in dashboard ─────────────────────────────────
resource "coder_app" "streamlit" {
  agent_id     = coder_agent.main.id
  slug         = "ddi-app"
  display_name = "DDI Predictor App"
  url          = "http://localhost:8501"
  icon         = "https://streamlit.io/images/brand/streamlit-mark-color.svg"
  subdomain    = false
  share        = "owner"

  healthcheck {
    url       = "http://localhost:8501/_stcore/health"
    interval  = 5
    threshold = 15
  }
}

# ── PRE-BUILT IMAGE — no custom build, pulls directly from Microsoft ──────────
resource "docker_image" "devcontainer" {
  name         = "mcr.microsoft.com/devcontainers/python:3.12"
  keep_locally = true
}

# ── PERSISTENT VOLUME ─────────────────────────────────────────────────────────
resource "docker_volume" "workspace" {
  name = "coder-${data.coder_workspace_owner.me.name}-ddi"
}

# ── WORKSPACE CONTAINER ───────────────────────────────────────────────────────
resource "docker_container" "workspace" {
  count   = data.coder_workspace.me.start_count
  image   = docker_image.devcontainer.image_id
  name    = "coder-${data.coder_workspace_owner.me.name}-${data.coder_workspace.me.name}"
  command = ["sh", "-c", coder_agent.main.init_script]

  env = [
    "CODER_AGENT_TOKEN=${coder_agent.main.token}"
  ]

  volumes {
    volume_name    = docker_volume.workspace.name
    container_path = "/workspaces/ddi"
  }

  restart = "unless-stopped"
}