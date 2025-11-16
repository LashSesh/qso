# Build stage
FROM rust:1.83-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY mef-core/Cargo.toml ./mef-core/
COPY mef-spiral/Cargo.toml ./mef-spiral/
COPY mef-ledger/Cargo.toml ./mef-ledger/
COPY mef-hdag/Cargo.toml ./mef-hdag/
COPY mef-ingestion/Cargo.toml ./mef-ingestion/
COPY mef-solvecoagula/Cargo.toml ./mef-solvecoagula/
COPY mef-tic/Cargo.toml ./mef-tic/
COPY mef-coupling/Cargo.toml ./mef-coupling/
COPY mef-audit/Cargo.toml ./mef-audit/
COPY mef-topology/Cargo.toml ./mef-topology/
COPY mef-api/Cargo.toml ./mef-api/
COPY mef-cli/Cargo.toml ./mef-cli/
COPY mef-storage/Cargo.toml ./mef-storage/
COPY mef-domains/Cargo.toml ./mef-domains/
COPY mef-vector-db/Cargo.toml ./mef-vector-db/
COPY mef-acquisition/Cargo.toml ./mef-acquisition/
COPY mef-specs/Cargo.toml ./mef-specs/
COPY mef-bench/Cargo.toml ./mef-bench/

# Create dummy source files to cache dependencies
RUN mkdir -p mef-core/src && echo "fn main() {}" > mef-core/src/lib.rs && \
    mkdir -p mef-spiral/src && echo "fn main() {}" > mef-spiral/src/lib.rs && \
    mkdir -p mef-ledger/src && echo "fn main() {}" > mef-ledger/src/lib.rs && \
    mkdir -p mef-hdag/src && echo "fn main() {}" > mef-hdag/src/lib.rs && \
    mkdir -p mef-ingestion/src && echo "fn main() {}" > mef-ingestion/src/lib.rs && \
    mkdir -p mef-solvecoagula/src && echo "fn main() {}" > mef-solvecoagula/src/lib.rs && \
    mkdir -p mef-tic/src && echo "fn main() {}" > mef-tic/src/lib.rs && \
    mkdir -p mef-coupling/src && echo "fn main() {}" > mef-coupling/src/lib.rs && \
    mkdir -p mef-audit/src && echo "fn main() {}" > mef-audit/src/lib.rs && \
    mkdir -p mef-topology/src && echo "fn main() {}" > mef-topology/src/lib.rs && \
    mkdir -p mef-api/src && echo "fn main() {}" > mef-api/src/lib.rs && \
    mkdir -p mef-api/src && echo "fn main() {}" > mef-api/src/main.rs && \
    mkdir -p mef-cli/src && echo "fn main() {}" > mef-cli/src/lib.rs && \
    mkdir -p mef-cli/src && echo "fn main() {}" > mef-cli/src/main.rs && \
    mkdir -p mef-storage/src && echo "fn main() {}" > mef-storage/src/lib.rs && \
    mkdir -p mef-domains/src && echo "fn main() {}" > mef-domains/src/lib.rs && \
    mkdir -p mef-vector-db/src && echo "fn main() {}" > mef-vector-db/src/lib.rs && \
    mkdir -p mef-acquisition/src && echo "fn main() {}" > mef-acquisition/src/lib.rs && \
    mkdir -p mef-specs/src && echo "fn main() {}" > mef-specs/src/lib.rs && \
    mkdir -p mef-bench/src && echo "fn main() {}" > mef-bench/src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --package mef-api

# Remove dummy files and copy real source
RUN rm -rf mef-*/src

# Copy source code
COPY mef-core ./mef-core
COPY mef-spiral ./mef-spiral
COPY mef-ledger ./mef-ledger
COPY mef-hdag ./mef-hdag
COPY mef-ingestion ./mef-ingestion
COPY mef-solvecoagula ./mef-solvecoagula
COPY mef-tic ./mef-tic
COPY mef-coupling ./mef-coupling
COPY mef-audit ./mef-audit
COPY mef-topology ./mef-topology
COPY mef-api ./mef-api
COPY mef-cli ./mef-cli
COPY mef-storage ./mef-storage
COPY mef-domains ./mef-domains
COPY mef-vector-db ./mef-vector-db
COPY mef-acquisition ./mef-acquisition
COPY mef-specs ./mef-specs
COPY mef-bench ./mef-bench

# Build application
RUN cargo build --release --package mef-api --bin mef-api

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mef && \
    mkdir -p /data && \
    chown -R mef:mef /data

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/mef-api /usr/local/bin/mef-api

# Set ownership
RUN chown mef:mef /usr/local/bin/mef-api

# Switch to non-root user
USER mef

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/mef-api"]
