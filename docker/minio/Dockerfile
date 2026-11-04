# syntax=docker/dockerfile:1
# Build the existing release from official sources: its registry image is unavailable.
FROM golang:1.22.7-bookworm AS builder

WORKDIR /src
ADD --checksum=sha256:72a80c9ef0bfe474653cbf9ce563b3691893dd5797ae4d1645b9e12f391b619e \
    https://codeload.github.com/minio/minio/tar.gz/refs/tags/RELEASE.2024-09-22T00-33-43Z /tmp/minio.tar.gz
RUN tar -xzf /tmp/minio.tar.gz --strip-components=1
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -tags kqueue \
    -ldflags="-s -w -X github.com/minio/minio/cmd.Version=2024-09-22T00:33:43Z -X github.com/minio/minio/cmd.ReleaseTag=RELEASE.2024-09-22T00-33-43Z" \
    -o /out/minio .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/minio /usr/bin/minio
COPY --from=builder /src/LICENSE /usr/share/licenses/minio/LICENSE
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
