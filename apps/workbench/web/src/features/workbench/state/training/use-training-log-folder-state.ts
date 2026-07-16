import { useCallback, useMemo, useState } from "react";
import { useLogExperimentsQuery } from "@/features/workbench/state/logs/use-log-queries";
import type { LogExperiment } from "@/lib/api/logs";

const LOG_FOLDER_RE = /^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$/;
const EMPTY_LOG_FOLDER_OPTIONS: LogExperiment[] = [];
const LOADING_LOG_FOLDERS_COPY = "Loading folders…";
const EMPTY_LOG_FOLDERS_COPY = "No safe experiment folders found";

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

function buildTrainingLogFolderView({
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
  const existingHelp = isLoading
    ? LOADING_LOG_FOLDERS_COPY
    : options.length === 0
      ? EMPTY_LOG_FOLDERS_COPY
      : "Select a top-level logs folder";

  return {
    value,
    isValid,
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
  const clearForConnectionChange = useCallback(() => {
    setMode("existing");
    setExistingValue("");
    setNewValue("");
  }, []);
  const state = useMemo(
    () => ({
      mode,
      existingValue,
      newValue,
      value: view.value,
      isValid: view.isValid,
      options,
      isLoading: logExperimentsQuery.isLoading,
      existingHelp: view.existingHelp,
      newValid: view.newValid,
      newError: view.newError,
    }),
    [
      existingValue,
      logExperimentsQuery.isLoading,
      mode,
      newValue,
      options,
      view.existingHelp,
      view.isValid,
      view.newError,
      view.newValid,
      view.value,
    ],
  );
  const actions = useMemo(
    () => ({
      selectMode: setMode,
      selectExisting: setExistingValue,
      nameNew: setNewValue,
    }),
    [],
  );

  return useMemo(
    () => ({ state, actions, clearForConnectionChange }),
    [actions, clearForConnectionChange, state],
  );
}
