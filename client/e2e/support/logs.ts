import type { Page } from "@playwright/test";
import { expect } from "./fixtures";

export async function logsControls(page: Page) {
  const heading = page.getByRole("heading", { name: /^(Actions|Logs)$/ });
  await expect(heading).toBeVisible();
  const actions = (await heading.textContent())?.trim() === "Actions";
  return {
    actions,
    search: actions
      ? page.getByRole("textbox", { name: "Search messages" })
      : page.getByPlaceholder("Search logs..."),
  };
}
