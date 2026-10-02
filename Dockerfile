# sensitive3d web service with the native OSGB bridge
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip cmake g++ make \
        openscenegraph libopenscenegraph-dev fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY native ./native
COPY scripts ./scripts
RUN ./scripts/build_bridge.sh

COPY pyproject.toml README.md ./
COPY sensitive3d ./sensitive3d
COPY web ./web
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install ".[yolo]"
ENV PATH=/opt/venv/bin:$PATH \
    S3D_OSGB_BRIDGE=/app/native/osgb_bridge/build/osgb_bridge

EXPOSE 8000
VOLUME ["/data"]
CMD ["sensitive3d", "serve", "--host", "0.0.0.0", "--port", "8000", "--workspace", "/data/workspace"]
