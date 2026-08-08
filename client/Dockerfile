# -----------------------------------------------------------------------------------
# Node-base — shared env vars for all targets
FROM node:22.14-alpine AS node-base

LABEL author.email="mitkojenia@gmail.com"
LABEL author.name="Eugene Mitsko"

# Important environment variables
ENV                                                                                 \
    # Root of the project
    ROOT_DIR="/code/project"                                                        \
    # Flag for docker for correct build
    CI=true                                                                          \
    # Vite API URL (overridden in docker-compose)
    VITE_API_URL=/api

# Set workdir
WORKDIR $ROOT_DIR

# Install core deps + pnpm
RUN                                                                                 \
    apk add --no-cache                                                              \
    git                                                                             \
    curl &&                                                                         \
    corepack enable &&                                                               \
    corepack prepare pnpm@9.15.9 --activate

# Copy package files
COPY package.json pnpm-lock.yaml .npmrc ./

# Install dependencies
RUN npm_config_minimum_release_age=0 pnpm install --frozen-lockfile

# -----------------------------------------------------------------------------------
# Build stage — compile frontend
FROM node-base AS build

# Copy source code
COPY . .

# Build the application
RUN pnpm run build

# -----------------------------------------------------------------------------------
# Serve stage — nginx for static files + reverse proxy
FROM nginx:1.27-alpine

RUN apk add --no-cache bash

COPY --from=build /code/project/dist /usr/share/nginx/html

ARG UPSTREAM_SERVER=server:8001
ENV UPSTREAM_SERVER=${UPSTREAM_SERVER}

# SPA fallback + /api reverse proxy to backend
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    location /api/ {\n\
        proxy_pass http://%s/api/;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
        proxy_buffering off;\n\
        proxy_read_timeout 600s;\n\
        proxy_send_timeout 600s;\n\
        chunked_transfer_encoding on;\n\
    }\n\
\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' "$UPSTREAM_SERVER" > /etc/nginx/conf.d/default.conf

EXPOSE 80
