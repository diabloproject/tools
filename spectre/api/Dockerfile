# Build stage
FROM rust:1.83-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache musl-dev

# Copy workspace files
COPY ../Cargo.toml ../Cargo.lock ./
COPY ../dsv ./dsv
COPY ../replay ./replay
COPY . ./api

# Build the application
WORKDIR /app/api
RUN cargo build --release

# Production stage
FROM alpine:latest AS runner

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache libgcc

# Copy binary from builder
COPY --from=builder /app/api/target/release/spectre-api ./

EXPOSE 8080

CMD ["./spectre-api"]
