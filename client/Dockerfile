# -----------------------------------------------------------------------------------
# Node-base — shared env vars for all targets
# Keep the Node major in sync with .github/workflows/lint.yml (client-test).
FROM node:24-alpine AS node-base

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

# Install core deps
RUN                                                                                 \
    apk add --no-cache                                                              \
    git                                                                             \
    curl

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# -----------------------------------------------------------------------------------
# Build stage — compile frontend
FROM node-base AS build

# Copy source code
COPY . .

# Build the application
RUN npm run build

# -----------------------------------------------------------------------------------
# Serve stage — nginx for static files + reverse proxy
# 1.27.3+ is required for runtime re-resolution of upstream replicas.
FROM nginx:1.27.5-alpine

RUN apk add --no-cache bash envsubst

COPY --from=build /code/project/dist /usr/share/nginx/html

ARG UPSTREAM_SERVER=server:8001
# Docker's embedded DNS; override for other orchestrators (e.g. kube-dns).
ARG NGINX_RESOLVER=127.0.0.11
ENV UPSTREAM_SERVER=${UPSTREAM_SERVER} \
    NGINX_RESOLVER=${NGINX_RESOLVER} \
    TRUSTED_PROXY_CIDR=127.0.0.1 \
    NGINX_ENVSUBST_FILTER="UPSTREAM_SERVER|TRUSTED_PROXY_CIDR|NGINX_RESOLVER"

COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE 80
