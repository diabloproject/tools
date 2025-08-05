FROM rust AS builder
COPY . .
RUN cargo build --release
FROM scratch
COPY --from=builder /target/release/tb /tb
CMD ["/tb"]
