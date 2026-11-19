import { defineConfig, devices } from "@playwright/test";

const baseURL = "http://127.0.0.1:13001";
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/support/global-setup.ts",
  fullyParallel: true,
  workers: 2,
  forbidOnly: !!process.env.CI,
  timeout: 45_000,
  expect: { timeout: 8_000 },
  reporter: [
    ["list"],
    ["html", { open: "never" }],
    ["json", { outputFile: "test-results/results.json" }],
  ],
  use: {
    baseURL,
    actionTimeout: 10_000,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    launchOptions: { executablePath },
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] }, testIgnore: "**/*mobile.spec.ts" },
    { name: "mobile", use: { ...devices["Pixel 7"] }, testMatch: "**/*mobile.spec.ts" },
  ],
  webServer: {
    command:
      "npm run dev -- --config e2e/vite.config.ts --host 127.0.0.1 --port 13001 --strictPort",
    url: baseURL,
    reuseExistingServer: false,
  },
});
