import { Providers } from "./providers";
import { ViewerApp } from "@/components/features/viewer/viewer-app";

export default function Page() {
  return (
    <Providers>
      <ViewerApp />
    </Providers>
  );
}
