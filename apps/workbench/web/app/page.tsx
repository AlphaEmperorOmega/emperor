import { Providers } from "./providers";
import { WorkbenchApp } from "@/features/workbench/components/workbench-app";
import { parseWorkbenchWorkspace } from "@/types/workbench";

export default async function Page({ searchParams }: PageProps<"/">) {
  const params = await searchParams;
  const workspaceValue = Array.isArray(params.workspace)
    ? params.workspace[0]
    : params.workspace;
  const initialWorkspace = parseWorkbenchWorkspace(workspaceValue);
  return (
    <Providers>
      <WorkbenchApp initialWorkspace={initialWorkspace} />
    </Providers>
  );
}
