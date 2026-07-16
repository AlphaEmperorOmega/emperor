export type WorkbenchWorkspace = "model" | "training" | "logs";

export function parseWorkbenchWorkspace(value: unknown): WorkbenchWorkspace {
  return value === "training" || value === "logs" ? value : "model";
}
