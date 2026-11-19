import { expect, request } from "@playwright/test";
import { admin, apiURL } from "./environment";

export default async function globalSetup() {
  const api = await request.newContext({ baseURL: apiURL, maxRedirects: 0 });
  try {
    const login = await api.post("/auth/login", { data: admin });
    expect(
      login.status(),
      "Dedicated rag-playwright admin login (see Compose status or rate limits)",
    ).toBe(200);
    const { access_token } = await login.json();
    const me = await api.get("/auth/me", { headers: { Authorization: `Bearer ${access_token}` } });
    await expect(me).toBeOK();
    expect(await me.json()).toMatchObject({ email: admin.email, role: "admin" });
    // The dedicated suite shares this admin and seeds 51 chunks for pagination.
    const writeLimit = await api.put("/admin/config/rate_limit_write_per_minute", {
      headers: { Authorization: `Bearer ${access_token}` },
      data: { value: "180" },
    });
    await expect(writeLimit).toBeOK();
    process.env.PLAYWRIGHT_ADMIN_TOKEN = access_token;
  } finally {
    await api.dispose();
  }
}
