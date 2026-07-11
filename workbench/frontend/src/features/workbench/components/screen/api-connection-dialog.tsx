import { type FormEvent, useEffect, useState } from "react";
import {
  Copy,
  KeyRound,
  LogOut,
  Plug,
  RefreshCw,
  RotateCcw,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { Input } from "@/components/ui/input";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import {
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { WORKBENCH_API_URL_ENV_NAME } from "@/lib/api/_connection-runtime";

const CORS_ENV_NAME = "WORKBENCH_API_CORS_ORIGINS";

function currentFrontendOrigin() {
  if (typeof window === "undefined") {
    return "";
  }
  return window.location.origin;
}

function Readout({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="grid gap-1 rounded-[8px] border border-line-soft bg-black/[0.18] px-3 py-2.5">
      <dt className="text-xs font-semibold text-ink-faint">{label}</dt>
      <dd className="min-w-0 break-all font-mono text-sm text-ink">{value}</dd>
    </div>
  );
}

function CopyableEnvValue({
  label,
  value,
  copyLabel,
}: {
  label: string;
  value: string;
  copyLabel: string;
}) {
  const { status, copy } = useCopyToClipboard(value);

  return (
    <div className="grid gap-2 rounded-[8px] border border-line-soft bg-black/[0.18] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-semibold text-ink-faint">{label}</div>
        <Button
          variant="secondary"
          onClick={copy}
          aria-label={copyLabel}
          className="h-8 px-2.5 text-xs"
        >
          <Copy className="h-3.5 w-3.5" aria-hidden />
          Copy
        </Button>
      </div>
      <textarea
        readOnly
        aria-label={label}
        value={value}
        rows={2}
        className="min-h-[64px] w-full resize-none rounded-[8px] border border-line bg-control-field px-3 py-2 font-mono text-[13px] leading-5 text-ink outline-none focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus"
      />
      <div
        role={status === "failed" ? "alert" : "status"}
        className="min-h-4 text-xs font-medium text-ink-faint"
      >
        {status === "copied" && "Copied"}
        {status === "failed" && "Clipboard copy failed"}
      </div>
    </div>
  );
}

export function ApiConnectionDialog({ onClose }: { onClose: () => void }) {
  const frontendOrigin = currentFrontendOrigin();
  const { connection, authentication, storage, actions } =
    useWorkbenchConnection();
  const apiBaseUrl = connection.apiBaseUrl;
  const [inputValue, setInputValue] = useState(apiBaseUrl);
  const [tokenInput, setTokenInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [authStorageError, setAuthStorageError] = useState<string | null>(null);
  const corsEnvValue = `${CORS_ENV_NAME}='${JSON.stringify([frontendOrigin])}'`;
  const frontendEnvValue = `${WORKBENCH_API_URL_ENV_NAME}=${apiBaseUrl}`;
  useEffect(() => setInputValue(apiBaseUrl), [apiBaseUrl]);

  const handleUse = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const outcome = await actions.useApiBaseUrl(inputValue);
    if (outcome.ok) {
      setInputValue(apiBaseUrl);
      setError(null);
    } else {
      setError(outcome.message);
    }
  };

  const handleReset = async () => {
    const outcome = await actions.resetApiBaseUrl();
    if (outcome.ok) {
      setInputValue(apiBaseUrl);
      setError(null);
    } else {
      setError(outcome.message);
    }
  };

  const handleAuth = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const token = tokenInput.trim();
    if (!token) {
      setAuthStorageError("Enter the bearer token supplied by the API operator.");
      return;
    }
    const outcome = await actions.signIn(token);
    if (!outcome.ok) {
      setAuthStorageError(outcome.message);
      return;
    }
    setTokenInput("");
    setAuthStorageError(null);
  };

  const handleLogout = async () => {
    const outcome = await actions.logout();
    if (!outcome.ok) {
      setAuthStorageError(outcome.message);
      return;
    }
    setTokenInput("");
    setAuthStorageError(null);
  };

  const handleRetry = async () => {
    setAuthStorageError(null);
    await actions.retry();
  };

  const canRetryAuthentication =
    authentication.state === "capability-failed" ||
    authentication.state === "protected-read-failed";

  const authStatus = authStorageError
    ? { role: "alert" as const, message: authStorageError, danger: true }
    : authentication.state === "capability-checking"
      ? {
          role: "status" as const,
          message: "Checking backend authentication requirements…",
          danger: false,
        }
      : authentication.state === "capability-failed"
        ? {
            role: "alert" as const,
            message:
              "Backend capabilities could not be read. Check the API connection and try again.",
            danger: true,
          }
        : authentication.state === "unauthenticated"
      ? {
          role: "alert" as const,
          message: "Authentication required. Enter the bearer token supplied by the API operator.",
          danger: true,
        }
      : authentication.state === "rejected"
        ? {
            role: "alert" as const,
            message: "The session bearer token was rejected. Replace it or log out.",
            danger: true,
          }
        : authentication.state === "protected-read-failed"
          ? {
              role: "alert" as const,
              message:
                "The session token is stored, but a protected API read failed. Check the API connection and try again.",
              danger: true,
            }
          : authentication.state === "checking"
            ? {
                role: "status" as const,
                message: "Checking the session bearer token…",
                danger: false,
              }
            : authentication.state === "authenticated"
              ? {
                  role: "status" as const,
                  message: "Authenticated for this browser session.",
                  danger: false,
                }
              : {
                  role: "status" as const,
                  message: "This backend does not require bearer authentication.",
                  danger: false,
                };

  return (
    <DialogShell
      titleId="api-connection-title"
      describedBy="api-connection-description"
      panelVariant="surface"
      size="md"
      onClose={onClose}
      panelClassName="overflow-hidden"
      header={
        <header className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <h2
              id="api-connection-title"
              className="flex items-center gap-2 text-base font-semibold text-ink"
            >
              <Plug className="h-4 w-4 text-violet" aria-hidden />
              API Connection
            </h2>
            <p
              id="api-connection-description"
              className="mt-1 max-w-[42rem] text-sm leading-6 text-ink-dim"
            >
              Choose the browser API base URL, manage session bearer
              authentication, and allow this Workbench origin in the backend
              CORS environment.
            </p>
          </div>
          <IconButton
            label="Close API connection settings"
            onClick={onClose}
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
          />
        </header>
      }
    >
      <div className="grid gap-4 overflow-y-auto p-4 sm:p-5">
        {connection.configurationError && (
          <div
            role="alert"
            className="rounded-[8px] border border-danger-line bg-danger-surface px-3 py-2.5 text-sm leading-6 text-danger-text"
          >
            {connection.configurationError}
          </div>
        )}
        {storage.message && (
          <div
            role="alert"
            className="rounded-[8px] border border-danger-line bg-danger-surface px-3 py-2.5 text-sm leading-6 text-danger-text"
          >
            {storage.message}
          </div>
        )}
        <div className="rounded-[8px] border border-amber/[0.25] bg-amber/[0.08] px-3 py-2.5 text-sm leading-6 text-ink-dim">
          CORS is enforced by the API server. Set{" "}
          <code className="font-mono text-ink">{CORS_ENV_NAME}</code> in the
          backend deployment; browser frontend code cannot add the required
          response headers.
        </div>

        <dl className="grid gap-3 sm:grid-cols-2">
          <Readout label="Current frontend origin" value={frontendOrigin} />
          <Readout label="Current API base URL" value={apiBaseUrl} />
          <Readout label="Backend authentication" value={authentication.mode} />
        </dl>

        <form
          className="grid gap-3 rounded-[8px] border border-line-soft bg-black/[0.18] p-3"
          onSubmit={handleAuth}
        >
          <label
            htmlFor="workbench-session-bearer-token"
            className="text-xs font-semibold text-ink-faint"
          >
            Session bearer token
          </label>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
            <Input
              id="workbench-session-bearer-token"
              type="password"
              autoComplete="current-password"
              spellCheck={false}
              value={tokenInput}
              onChange={(event) => {
                setTokenInput(event.target.value);
                if (authStorageError) {
                  setAuthStorageError(null);
                }
              }}
              placeholder={authentication.hasToken ? "Enter replacement token" : "Enter token"}
            />
            <Button
              variant="primary"
              type="submit"
              className="h-9"
              disabled={!tokenInput.trim()}
            >
              <KeyRound className="h-3.5 w-3.5" aria-hidden />
              {authentication.hasToken ? "Replace token" : "Sign in"}
            </Button>
            {authentication.hasToken && (
              <Button
                variant="secondary"
                type="button"
                className="h-9"
                onClick={handleLogout}
              >
                <LogOut className="h-3.5 w-3.5" aria-hidden />
                Log out
              </Button>
            )}
          </div>
          <div
            role={authStatus.role}
            className={
              authStatus.danger
                ? "min-h-4 text-xs font-medium text-danger-text"
                : "min-h-4 text-xs font-medium text-ink-faint"
            }
          >
            {authStatus.message}
          </div>
          {canRetryAuthentication && (
            <Button
              variant="secondary"
              type="button"
              className="h-9 justify-self-start"
              onClick={handleRetry}
              disabled={connection.isChanging}
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden />
              Try again
            </Button>
          )}
          <p className="text-xs leading-5 text-ink-faint">
            The token is kept only in this tab&apos;s browser session storage. It
            is removed when the session ends and is never read from a public
            build environment variable.
          </p>
        </form>

        <form
          className="grid gap-3 rounded-[8px] border border-line-soft bg-black/[0.18] p-3"
          onSubmit={handleUse}
        >
          <label
            htmlFor="workbench-api-base-url"
            className="text-xs font-semibold text-ink-faint"
          >
            API base URL
          </label>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
            <Input
              id="workbench-api-base-url"
              value={inputValue}
              onChange={(event) => {
                setInputValue(event.target.value);
                if (error) {
                  setError(null);
                }
              }}
              aria-invalid={Boolean(error)}
              aria-describedby={error ? "workbench-api-base-url-error" : undefined}
              placeholder="Configured default API URL"
            />
            <Button variant="primary" type="submit" className="h-9">
              <Plug className="h-3.5 w-3.5" aria-hidden />
              Use
            </Button>
            <Button
              variant="secondary"
              type="button"
              className="h-9"
              onClick={handleReset}
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              Reset
            </Button>
          </div>
          <div
            id="workbench-api-base-url-error"
            role={error ? "alert" : undefined}
            className="min-h-4 text-xs font-medium text-danger-text"
          >
            {error}
          </div>
        </form>

        <div className="grid gap-3">
          <CopyableEnvValue
            label="Backend CORS environment variable"
            value={corsEnvValue}
            copyLabel="Copy backend CORS environment variable"
          />
          <CopyableEnvValue
            label="Frontend API URL environment variable"
            value={frontendEnvValue}
            copyLabel="Copy frontend API URL environment variable"
          />
        </div>

        <p className="text-xs leading-5 text-ink-faint">
          If bearer auth is enabled, keep token verification in the backend
          environment too. The browser can attach a session bearer token, but it
          does not own the backend auth mode or token secret.
        </p>
        <p className="text-xs leading-5 text-ink-faint">
          Hosted or read-only backends can leave local mutation endpoints
          disabled. Training, log deletion, and config snapshots require
          backend-side unsafe local mutation opt-in with{" "}
          <code className="font-mono text-ink">
            WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS=true
          </code>{" "}
          only when those local mutations are intentionally allowed.
        </p>
      </div>
    </DialogShell>
  );
}
