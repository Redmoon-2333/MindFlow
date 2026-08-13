import { createRoot } from "react-dom/client";
import App from "./App.tsx";

// Bootstrap ticket exchange is handled inside <App /> so the login screen can
// show an in-progress state and keep the one-time ticket in the URL on failure
// (retryable). Rendering immediately also avoids a blank page while waiting.
createRoot(document.getElementById("root")!).render(<App />);
