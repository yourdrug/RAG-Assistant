import axios from "axios";
import { useAuthStore } from "@/stores/auth-store";

const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

function generateIdempotencyKey(): string {
  return crypto.randomUUID();
}

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  // Auto-add Idempotency-Key for write operations (POST, PUT, PATCH, DELETE)
  const method = config.method?.toUpperCase();
  if (method && ["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
    if (!config.headers["Idempotency-Key"]) {
      config.headers["Idempotency-Key"] = generateIdempotencyKey();
    }
  }
  return config;
});

let _queryClientRef: { clear: () => void } | null = null;

export function setQueryClientRef(qc: { clear: () => void }) {
  _queryClientRef = qc;
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      _queryClientRef?.clear();
      useAuthStore.getState().logout();
      window.location.href = "/login";
    }
    return Promise.reject(error);
  },
);
