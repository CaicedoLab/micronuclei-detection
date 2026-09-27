FROM ghcr.io/prefix-dev/pixi:latest
WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --locked
ENV PATH=/app/.pixi/envs/default/bin:$PATH