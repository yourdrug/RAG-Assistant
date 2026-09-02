"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { setQueryClientRef } from "@/shared/api/client";

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [qc] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { staleTime: 60_000, retry: 1, refetchOnWindowFocus: false } },
      }),
  );

  useEffect(() => {
    setQueryClientRef(qc);
    return () => setQueryClientRef(null as unknown as { clear: () => void });
  }, [qc]);

  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}
