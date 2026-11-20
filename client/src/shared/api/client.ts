import axios, { isAxiosError } from "axios";
import toast from "react-hot-toast";
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
  if (token && !config.headers.has("Authorization") && config.url !== "/auth/login") {
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

export function setQueryClientRef(qc: { clear: () => void } | null) {
  _queryClientRef = qc;
}

export function expireSession(token: string) {
  const auth = useAuthStore.getState();
  if (auth.token !== token) return;
  _queryClientRef?.clear();
  auth.logout();
  toast.error("Your session has expired. Please sign in again.", { id: "session-expired" });
}

apiClient.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (isAxiosError(error) && error.response?.status === 401) {
      const authorization = error.config?.headers.get("Authorization");
      if (typeof authorization === "string" && authorization.startsWith("Bearer ")) {
        expireSession(authorization.slice(7));
      }
    }
    return Promise.reject(error);
  },
);
