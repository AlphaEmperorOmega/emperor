import { describe, expect, it } from "vitest";
import {
  buildTrainingLogFolderView,
} from "@/features/workbench/state/training/use-training-log-folder-state";

const logOptions = [
  { experiment: "baseline", runCount: 3, relativePath: "baseline" },
];

describe("training log folder state", () => {
  it("validates an existing log folder against loaded experiments", () => {
    const view = buildTrainingLogFolderView({
      mode: "existing",
      existingValue: "baseline",
      newValue: "",
      options: logOptions,
      isLoading: false,
    });

    expect(view).toMatchObject({
      value: "baseline",
      isValid: true,
      label: "logs/baseline",
      existingHelp: "Select a top-level logs folder",
    });
  });

  it("rejects unknown existing folders and invalid new folder names", () => {
    expect(
      buildTrainingLogFolderView({
        mode: "existing",
        existingValue: "missing",
        newValue: "",
        options: logOptions,
        isLoading: false,
      }),
    ).toMatchObject({
      value: "missing",
      isValid: false,
      label: "Choose log folder",
    });

    expect(
      buildTrainingLogFolderView({
        mode: "new",
        existingValue: "",
        newValue: "bad-name",
        options: logOptions,
        isLoading: false,
      }),
    ).toMatchObject({
      value: "bad-name",
      isValid: false,
      newValid: false,
      newError: "Use letters and numbers separated by single underscores.",
    });
  });

  it("accepts underscore-separated new folder names", () => {
    expect(
      buildTrainingLogFolderView({
        mode: "new",
        existingValue: "",
        newValue: "fresh_run_2",
        options: logOptions,
        isLoading: false,
      }),
    ).toMatchObject({
      value: "fresh_run_2",
      isValid: true,
      label: "logs/fresh_run_2",
      newValid: true,
      newError: "",
    });
  });
});
