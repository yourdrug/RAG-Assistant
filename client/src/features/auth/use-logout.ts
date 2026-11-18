import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/auth-store";

export function useLogout() {
  const queryClient = useQueryClient();
  const logout = useAuthStore((state) => state.logout);
  const navigate = useNavigate();

  return () => {
    queryClient.clear();
    logout();
    navigate("/login");
  };
}
