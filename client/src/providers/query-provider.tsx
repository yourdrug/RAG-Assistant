"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { setQueryClientRef } from "@/shared/api/client";

import { shouldRetryRequest } from "@/shared/lib/errors";

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [qc] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 60_000, retry: shouldRetryRequest, refetchOnWindowFocus: false },
        },
      }),
  );

  useEffect(() => {
    setQueryClientRef(qc);
    return () => setQueryClientRef(null);
  }, [qc]);

  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}
