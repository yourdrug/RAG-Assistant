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

RUN apk add --no-cache bash envsubst

COPY --from=build /code/project/dist /usr/share/nginx/html

ARG UPSTREAM_SERVER=server:8001
ENV UPSTREAM_SERVER=${UPSTREAM_SERVER}

COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE 80
