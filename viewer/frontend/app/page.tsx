import { Providers } from "./providers";
import { ViewerApp } from "@/features/viewer/components/viewer-app";

export default function Page() {
  return (
    <Providers>
      <ViewerApp />
    </Providers>
  );
}
