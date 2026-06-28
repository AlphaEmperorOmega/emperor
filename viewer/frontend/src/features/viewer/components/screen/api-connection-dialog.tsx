import { type FormEvent, useState } from "react";
import { Copy, Plug, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { Input } from "@/components/ui/input";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { useApiConnection } from "@/features/viewer/providers/viewer-providers";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import {
  normalizeViewerApiBaseUrl,
  VIEWER_API_BASE_URL,
  VIEWER_API_URL_ENV_NAME,
} from "@/lib/api";

const CORS_ENV_NAME = "VIEWER_API_CORS_ORIGINS";

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
  const {
    apiBaseUrl,
    setApiBaseUrl: switchApiBaseUrl,
    resetApiBaseUrl,
  } = useApiConnection();
  const [inputValue, setInputValue] = useState(apiBaseUrl);
  const [error, setError] = useState<string | null>(null);
  const corsEnvValue = `${CORS_ENV_NAME}='${JSON.stringify([frontendOrigin])}'`;
  const frontendEnvValue = `${VIEWER_API_URL_ENV_NAME}=${apiBaseUrl}`;

  const handleUse = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedUrl = normalizeViewerApiBaseUrl(inputValue);
    if (!normalizedUrl) {
      setError(
        "Enter an absolute http:// or https:// URL without a query string or fragment.",
      );
      return;
    }
    try {
      const nextApiBaseUrl = switchApiBaseUrl(normalizedUrl);
      setInputValue(nextApiBaseUrl);
      setError(null);
    } catch (errorValue) {
      setError(
        errorValue instanceof Error
          ? errorValue.message
          : "This API base URL is not allowed by this build.",
      );
    }
  };

  const handleReset = () => {
    const nextApiBaseUrl = resetApiBaseUrl();
    setInputValue(nextApiBaseUrl);
    setError(null);
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
              Choose the browser API base URL and allow this Viewer origin in
              the backend CORS environment.
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
        <div className="rounded-[8px] border border-amber/[0.25] bg-amber/[0.08] px-3 py-2.5 text-sm leading-6 text-ink-dim">
          CORS is enforced by the API server. Set{" "}
          <code className="font-mono text-ink">{CORS_ENV_NAME}</code> in the
          backend deployment; browser frontend code cannot add the required
          response headers.
        </div>

        <dl className="grid gap-3 sm:grid-cols-2">
          <Readout label="Current frontend origin" value={frontendOrigin} />
          <Readout label="Current API base URL" value={apiBaseUrl} />
        </dl>

        <form
          className="grid gap-3 rounded-[8px] border border-line-soft bg-black/[0.18] p-3"
          onSubmit={handleUse}
        >
          <label
            htmlFor="viewer-api-base-url"
            className="text-xs font-semibold text-ink-faint"
          >
            API base URL
          </label>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
            <Input
              id="viewer-api-base-url"
              value={inputValue}
              onChange={(event) => {
                setInputValue(event.target.value);
                if (error) {
                  setError(null);
                }
              }}
              aria-invalid={Boolean(error)}
              aria-describedby={error ? "viewer-api-base-url-error" : undefined}
              placeholder={VIEWER_API_BASE_URL}
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
            id="viewer-api-base-url-error"
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
            VIEWER_API_ALLOW_UNSAFE_LOCAL_MUTATIONS=true
          </code>{" "}
          only when those local mutations are intentionally allowed.
        </p>
      </div>
    </DialogShell>
  );
}
