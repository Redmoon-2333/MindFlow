import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

/** Top-level error boundary (architecture review 💡18): a render error in any
 *  page must not blank out the whole app. Shows a recoverable error card
 *  instead of unmounting the React tree. */
export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, message: "" };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error.message || "未知错误" };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Keep console visibility for debugging without crashing the render.
    console.error("MindFlow render error:", error, info.componentStack);
  }

  private handleReload = (): void => {
    this.setState({ hasError: false, message: "" });
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          minHeight: "100vh", background: "var(--color-bg)",
        }}>
          <div className="card" style={{ width: 440, padding: 40, textAlign: "center" }}>
            <h1 style={{ fontSize: 24, color: "var(--color-primary)", marginBottom: 8 }}>
              MindFlow
            </h1>
            <p style={{ color: "var(--color-text-secondary)", marginBottom: 16 }}>
              页面渲染出现问题，请刷新重试
            </p>
            <div className="error-box" style={{ textAlign: "left", marginBottom: 16 }}>
              {this.state.message}
            </div>
            <button className="btn btn-primary" onClick={this.handleReload}>
              刷新页面
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
