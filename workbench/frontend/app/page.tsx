import { Providers } from "./providers";
import { WorkbenchApp } from "@/features/workbench/components/workbench-app";

export default function Page() {
  return (
    <Providers>
      <WorkbenchApp />
    </Providers>
  );
}
