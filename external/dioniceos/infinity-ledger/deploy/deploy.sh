#!/bin/bash
# Deployment automation script for MEF API
# Usage: ./deploy/deploy.sh [environment] [version]
#   environment: dev, staging, or production
#   version: git tag or branch name (default: current branch)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/lashsesh/infinityledger}"

# Function to print colored messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Function to check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"
    command -v git >/dev/null 2>&1 || error "Git is not installed"
    
    info "All prerequisites met"
}

# Function to validate environment
validate_environment() {
    local env=$1
    
    if [[ ! "$env" =~ ^(dev|staging|production)$ ]]; then
        error "Invalid environment: $env. Must be dev, staging, or production"
    fi
    
    info "Deploying to environment: $env"
}

# Function to build Docker images
build_images() {
    local version=$1
    
    info "Building Docker images (version: $version)..."
    
    cd "$PROJECT_ROOT"
    
    # Build the API image
    docker build \
        -t "${DOCKER_REGISTRY}/mef-api:${version}" \
        -t "${DOCKER_REGISTRY}/mef-api:latest" \
        -f Dockerfile \
        .
    
    info "Docker images built successfully"
}

# Function to run tests before deployment
run_tests() {
    info "Running tests before deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Run Rust tests
    cargo test --workspace --release || error "Tests failed"
    
    # Run integration tests
    cargo test --test integration_* --release || warn "Some integration tests failed"
    
    info "Tests completed"
}

# Function to push Docker images
push_images() {
    local version=$1
    
    info "Pushing Docker images to registry..."
    
    docker push "${DOCKER_REGISTRY}/mef-api:${version}"
    docker push "${DOCKER_REGISTRY}/mef-api:latest"
    
    info "Docker images pushed successfully"
}

# Function to deploy to environment
deploy() {
    local env=$1
    local version=$2
    
    info "Deploying version $version to $env..."
    
    cd "$PROJECT_ROOT"
    
    case "$env" in
        dev)
            deploy_dev "$version"
            ;;
        staging)
            deploy_staging "$version"
            ;;
        production)
            deploy_production "$version"
            ;;
    esac
    
    info "Deployment completed"
}

# Function to deploy to dev environment
deploy_dev() {
    local version=$1
    
    info "Starting dev environment..."
    
    export MEF_VERSION="$version"
    docker-compose -f docker-compose.yml --profile dev up -d
    
    # Wait for services to be healthy
    wait_for_health "http://localhost:8080/healthz"
    
    info "Dev environment is up and running"
}

# Function to deploy to staging environment
deploy_staging() {
    local version=$1
    
    info "Starting staging environment..."
    
    export MEF_VERSION="$version"
    export ENVIRONMENT="staging"
    docker-compose \
        -f infinity-ledger-main/docker-compose.production.yml \
        --profile staging \
        up -d
    
    # Wait for services to be healthy
    wait_for_health "http://localhost:8080/healthz"
    
    info "Staging environment is up and running"
}

# Function to deploy to production environment
deploy_production() {
    local version=$1
    
    warn "Deploying to PRODUCTION environment"
    read -p "Are you sure you want to proceed? (yes/no): " -r
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        error "Production deployment cancelled"
    fi
    
    info "Starting production environment..."
    
    export MEF_VERSION="$version"
    export ENVIRONMENT="production"
    docker-compose \
        -f infinity-ledger-main/docker-compose.production.yml \
        --profile production \
        --profile monitoring \
        up -d
    
    # Wait for services to be healthy
    wait_for_health "http://localhost:8080/healthz"
    
    info "Production environment is up and running"
    info "Monitoring available at http://localhost:3000 (Grafana)"
    info "Metrics available at http://localhost:9090 (Prometheus)"
}

# Function to wait for service health
wait_for_health() {
    local health_url=$1
    local max_attempts=30
    local attempt=1
    
    info "Waiting for service to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$health_url" >/dev/null 2>&1; then
            info "Service is healthy"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "Service failed to become healthy after $max_attempts attempts"
}

# Function to rollback deployment
rollback() {
    local env=$1
    
    warn "Rolling back deployment in $env environment..."
    
    cd "$PROJECT_ROOT"
    
    case "$env" in
        dev)
            docker-compose -f docker-compose.yml --profile dev down
            ;;
        staging)
            docker-compose -f infinity-ledger-main/docker-compose.production.yml --profile staging down
            ;;
        production)
            docker-compose -f infinity-ledger-main/docker-compose.production.yml --profile production down
            ;;
    esac
    
    info "Rollback completed"
}

# Function to show deployment status
show_status() {
    local env=$1
    
    info "Deployment status for $env environment:"
    
    docker-compose ps
    
    info "Logs:"
    docker-compose logs --tail=50 api
}

# Main function
main() {
    local environment="${1:-dev}"
    local version="${2:-$(git rev-parse --abbrev-ref HEAD)}"
    
    info "MEF Deployment Script"
    info "====================="
    
    check_prerequisites
    validate_environment "$environment"
    
    # Get current git commit hash
    local git_commit=$(git rev-parse --short HEAD)
    version="${version}-${git_commit}"
    
    info "Version to deploy: $version"
    
    # Build and test
    build_images "$version"
    run_tests
    
    # Push images (skip for dev environment)
    if [[ "$environment" != "dev" ]]; then
        push_images "$version"
    fi
    
    # Deploy
    deploy "$environment" "$version"
    
    # Show status
    show_status "$environment"
    
    info "Deployment successful!"
    info "API endpoint: http://localhost:8080"
}

# Handle command line arguments
case "${1:-}" in
    rollback)
        rollback "${2:-dev}"
        ;;
    status)
        show_status "${2:-dev}"
        ;;
    *)
        main "$@"
        ;;
esac
