export function virtualenvPython(repositoryRoot: string, platform?: string): string;
export function pythonCommand(platform?: string): string;
export function resolveRepositoryPython(
  repositoryRoot: string,
  options?: {
    environment?: NodeJS.ProcessEnv;
    platform?: string;
  },
): string;
export function nextCli(frontendRoot: string): string;
export function matplotlibConfigDirectory(): string;
