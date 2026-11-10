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
        <div className="error-boundary">
          <div className="card error-boundary-card">
            <h1 className="error-boundary-title">MindFlow</h1>
            <p className="error-boundary-text">页面渲染出现问题，请刷新重试</p>
            {/* The raw message is for diagnosis, not for the reader: it is
                collapsed by default and never in the primary copy (§8.21.3). */}
            <details className="error-boundary-details">
              <summary>技术详情</summary>
              <p className="error-boundary-raw">{this.state.message}</p>
            </details>
            <button className="btn btn-primary error-boundary-action" onClick={this.handleReload}>
              刷新页面
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
