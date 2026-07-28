import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import { bootstrapFromFragment } from "./api";

async function start(): Promise<void> {
  try {
    await bootstrapFromFragment();
  } catch (error) {
    console.error("MindFlow bootstrap failed", error);
  }
  createRoot(document.getElementById("root")!).render(<App />);
}

void start();
