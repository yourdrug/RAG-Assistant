import { mergeConfig } from "vite";
import appConfig from "../vite.config";

export default mergeConfig(appConfig, {
  define: { "import.meta.env.VITE_API_URL": JSON.stringify("/api") },
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:18001" },
    },
  },
});
