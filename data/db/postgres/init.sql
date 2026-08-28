-- init.sql — create database and user on first postgres start.
-- Schema is managed by Alembic (alembic upgrade head runs in entrypoint.sh).

CREATE DATABASE IF NOT EXISTS ragdb;

CREATE USER IF NOT EXISTS raguser WITH PASSWORD 'ragpassword';

GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;

\c ragdb
GRANT ALL ON SCHEMA public TO raguser;
