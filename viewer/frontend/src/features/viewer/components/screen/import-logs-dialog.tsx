import {
  type ChangeEvent,
  type FormEvent,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileArchive,
  LoaderCircle,
  Upload,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { useTargetConfig } from "@/features/viewer/providers/viewer-providers";
import { useLogQueryCache } from "@/features/viewer/state/logs/use-log-query-cache";
import { importLogArchive, type LogArchiveImportResponse } from "@/lib/api";

function formatFileSize(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = size / 1024;
  for (const unit of units) {
    if (value < 1024 || unit === "GB") {
      return `${value.toFixed(value < 10 ? 1 : 0).replace(/\.0$/, "")} ${unit}`;
    }
    value /= 1024;
  }
  return `${size} B`;
}

const logImportsDisabledMessage = "Log imports are disabled by this backend.";
const logImportsDisabledHint =
  "For local project uploads, use the local backend defaults or set VIEWER_API_ALLOW_LOG_IMPORTS=true before starting the backend.";

function messageFromImportError(error: unknown) {
  if (!(error instanceof Error)) {
    return "Log import failed.";
  }
  if (error.message.includes("Local mutation endpoints are disabled")) {
    return logImportsDisabledMessage;
  }
  if (
    error.name === "TypeError" ||
    error.message.toLowerCase().includes("failed to fetch") ||
    error.message.toLowerCase().includes("network")
  ) {
    return "Network failure while uploading the log archive.";
  }
  return error.message;
}

function successMessage(result: LogArchiveImportResponse) {
  const imported =
    result.extractedFileCount === 1
      ? "1 file imported"
      : `${result.extractedFileCount} files imported`;
  if (result.skippedFileCount === 0) {
    return `${imported}.`;
  }
  const skipped =
    result.skippedFileCount === 1
      ? "1 file skipped"
      : `${result.skippedFileCount} files skipped`;
  return `${imported}; ${skipped}.`;
}

export function ImportLogsDialog({ onClose }: { onClose: () => void }) {
  const { capabilities } = useTargetConfig();
  const logQueryCache = useLogQueryCache();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LogArchiveImportResponse | null>(null);
  const uploadsEnabled = capabilities.uploadsEnabled;
  const maxUploadSize = capabilities.maxUploadSize;

  const selectedFileTooLarge = useMemo(
    () =>
      selectedFile && maxUploadSize !== null
        ? selectedFile.size > maxUploadSize
        : false,
    [maxUploadSize, selectedFile],
  );
  const canImport =
    uploadsEnabled && selectedFile !== null && !selectedFileTooLarge && !isImporting;

  const handleChooseFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    if (file && !file.name.toLowerCase().endsWith(".zip")) {
      setError("Choose a .zip file created by download_logs.sh.");
      return;
    }
    if (file && maxUploadSize !== null && file.size > maxUploadSize) {
      setError(
        `Selected file is larger than the ${formatFileSize(maxUploadSize)} limit.`,
      );
      return;
    }
    setError(null);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!uploadsEnabled) {
      setError(logImportsDisabledMessage);
      return;
    }
    if (!selectedFile) {
      setError("Choose a .zip file to import.");
      return;
    }
    if (!selectedFile.name.toLowerCase().endsWith(".zip")) {
      setError("Choose a .zip file created by download_logs.sh.");
      return;
    }
    if (selectedFileTooLarge) {
      setError(
        maxUploadSize === null
          ? "Selected file is too large."
          : `Selected file is larger than the ${formatFileSize(maxUploadSize)} limit.`,
      );
      return;
    }

    setIsImporting(true);
    setError(null);
    setResult(null);
    try {
      const importResult = await importLogArchive(selectedFile);
      setResult(importResult);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      void logQueryCache.refreshAfterMutation();
    } catch (caughtError) {
      setError(messageFromImportError(caughtError));
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <DialogShell
      titleId="import-logs-title"
      describedBy="import-logs-description"
      panelVariant="surface"
      size="md"
      onClose={isImporting ? undefined : onClose}
      closeOnEscape={!isImporting}
      panelClassName="overflow-hidden"
      header={
        <header className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <h2
              id="import-logs-title"
              className="flex items-center gap-2 text-base font-semibold text-ink"
            >
              <FileArchive className="h-4 w-4 text-violet" aria-hidden />
              Import Logs
            </h2>
            <p
              id="import-logs-description"
              className="mt-1 max-w-[42rem] text-sm leading-6 text-ink-dim"
            >
              Upload a zip produced by download_logs.sh and extract it into the
              backend logs folder.
            </p>
          </div>
          <IconButton
            label="Close import logs dialog"
            onClick={onClose}
            disabled={isImporting}
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
          />
        </header>
      }
    >
      <form className="grid gap-4 overflow-y-auto p-4 sm:p-5" onSubmit={handleSubmit}>
        {!uploadsEnabled && (
          <div className="rounded-[8px] border border-amber/[0.25] bg-amber/[0.08] px-3 py-2.5 text-sm leading-6 text-ink-dim">
            <p>{logImportsDisabledMessage}</p>
            <p className="mt-1 text-xs text-ink-faint">
              {logImportsDisabledHint}
            </p>
          </div>
        )}

        <div className="grid gap-3 rounded-[8px] border border-line-soft bg-black/[0.18] p-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
          <label
            id="log-archive-file-label"
            htmlFor="log-archive-file"
            className="text-xs font-semibold text-ink-faint"
          >
            Log archive zip file
          </label>
          <Button
            variant="secondary"
            onClick={handleChooseFile}
            disabled={isImporting || !uploadsEnabled}
            aria-describedby="log-archive-file-label"
            className="w-full sm:w-auto"
          >
            <FileArchive className="h-3.5 w-3.5" aria-hidden />
            Choose Zip
          </Button>
          <input
            id="log-archive-file"
            ref={fileInputRef}
            type="file"
            accept=".zip,application/zip,application/x-zip-compressed"
            disabled={isImporting || !uploadsEnabled}
            onChange={handleFileChange}
            tabIndex={-1}
            className="sr-only"
          />
        </div>

        <div className="rounded-[8px] border border-line-soft bg-black/[0.18] px-3 py-2.5">
          <div className="text-xs font-semibold text-ink-faint">Selected file</div>
          <div className="mt-1 min-h-5 break-all text-sm text-ink">
            {selectedFile ? selectedFile.name : "No file selected"}
          </div>
          <div className="mt-1 text-xs font-medium text-ink-faint">
            {selectedFile ? formatFileSize(selectedFile.size) : "Select a .zip archive"}
          </div>
        </div>

        {result && (
          <div
            role="status"
            className="flex items-start gap-2 rounded-[8px] border border-ok/[0.28] bg-ok/[0.08] px-3 py-2.5 text-sm leading-6 text-ink-dim"
          >
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-ok" aria-hidden />
            <span>
              {successMessage(result)} Destination:{" "}
              <code className="font-mono text-ink">{result.destinationRoot}</code>
            </span>
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-[8px] border border-danger-line bg-danger-soft px-3 py-2.5 text-sm leading-6 text-danger-text"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>{error}</span>
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2 border-t border-line-soft pt-4">
          <Button variant="secondary" onClick={onClose} disabled={isImporting}>
            Cancel
          </Button>
          <Button variant="primary" type="submit" disabled={!canImport}>
            {isImporting ? (
              <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden />
            ) : (
              <Upload className="h-3.5 w-3.5" aria-hidden />
            )}
            {isImporting ? "Importing..." : "Import Logs"}
          </Button>
        </div>
      </form>
    </DialogShell>
  );
}
