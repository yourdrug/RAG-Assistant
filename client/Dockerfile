# Stage 1: Build
FROM node:22.11-alpine AS build

ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci
RUN npm install --no-save @rolldown/binding-linux-x64-musl

COPY . .
RUN npm run build

# Stage 2: Serve
FROM nginx:1.27-alpine

RUN apk add --no-cache bash

COPY --from=build /app/dist /usr/share/nginx/html

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
