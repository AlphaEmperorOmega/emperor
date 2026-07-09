import { useMemo, useState } from "react";
import { workbenchStatusCopy } from "@/features/workbench/components/shared/status-copy";
import { useLogExperimentsQuery } from "@/features/workbench/state/logs/use-log-queries";
import { type LogExperiment } from "@/lib/api";

const LOG_FOLDER_RE = /^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$/;
const EMPTY_LOG_FOLDER_OPTIONS: LogExperiment[] = [];

export type LogFolderMode = "existing" | "new";

type TrainingLogFolderStateInput = {
  enabled?: boolean;
};

type TrainingLogFolderViewInput = {
  mode: LogFolderMode;
  existingValue: string;
  newValue: string;
  options: LogExperiment[];
  isLoading: boolean;
};

export function buildTrainingLogFolderView({
  mode,
  existingValue,
  newValue,
  options,
  isLoading,
}: TrainingLogFolderViewInput) {
  const existingValid = Boolean(
    existingValue && options.some((option) => option.experiment === existingValue),
  );
  const newValid = LOG_FOLDER_RE.test(newValue);
  const newError =
    newValue.length === 0
      ? "Enter a folder name."
      : newValid
        ? ""
        : "Use letters and numbers separated by single underscores.";
  const value = mode === "existing" ? existingValue : newValue;
  const isValid = mode === "existing" ? existingValid : newValid;
  const label = isValid ? `logs/${value}` : "Choose log folder";
  const existingHelp = isLoading
    ? workbenchStatusCopy.loading.logFolders
    : options.length === 0
      ? workbenchStatusCopy.empty.safeLogFolders
      : "Select a top-level logs folder";

  return {
    value,
    isValid,
    label,
    existingHelp,
    newValid,
    newError,
  };
}

export function useTrainingLogFolderState({
  enabled = true,
}: TrainingLogFolderStateInput = {}) {
  const [mode, setMode] = useState<LogFolderMode>("existing");
  const [existingValue, setExistingValue] = useState("");
  const [newValue, setNewValue] = useState("");
  const logExperimentsQuery = useLogExperimentsQuery({
    enabled: enabled && mode === "existing",
  });
  const options = logExperimentsQuery.data?.experiments ?? EMPTY_LOG_FOLDER_OPTIONS;
  const view = useMemo(
    () =>
      buildTrainingLogFolderView({
        mode,
        existingValue,
        newValue,
        options,
        isLoading: logExperimentsQuery.isLoading,
      }),
    [existingValue, logExperimentsQuery.isLoading, mode, newValue, options],
  );

  return {
    mode,
    setMode,
    existingValue,
    setExistingValue,
    newValue,
    setNewValue,
    options,
    isLoading: logExperimentsQuery.isLoading,
    existingHelp: view.existingHelp,
    newValid: view.newValid,
    newError: view.newError,
    value: view.value,
    isValid: view.isValid,
    label: view.label,
  };
}
