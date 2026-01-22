import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Uncaught error:', error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            if (typeof this.props.fallback === 'function') {
                return this.props.fallback(this.state.error || new Error('Unknown error'));
            }
            if (this.props.fallback) return this.props.fallback;

            return (
                <div style={{ padding: '20px', color: 'red', background: 'rgba(0,0,0,0.8)', borderRadius: '8px' }}>
                    <h3>Something went wrong.</h3>
                    <pre style={{ whiteSpace: 'pre-wrap', fontSize: '10px' }}>
                        {this.state.error?.message}
                    </pre>
                </div>
            );
        }

        return this.props.children;
    }
}
