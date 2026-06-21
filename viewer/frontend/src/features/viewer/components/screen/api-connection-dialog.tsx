import { Copy, Plug, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import {
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
  const apiBaseUrl = VIEWER_API_BASE_URL;
  const corsEnvValue = `${CORS_ENV_NAME}='${JSON.stringify([frontendOrigin])}'`;
  const frontendEnvValue = `${VIEWER_API_URL_ENV_NAME}=${apiBaseUrl}`;

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
              Configure the browser API URL in the frontend environment and allow
              this Viewer origin in the backend CORS environment.
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
