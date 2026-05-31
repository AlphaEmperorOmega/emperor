import { Providers } from "./providers";
import { ViewerApp } from "@/components/viewer-app";

export default function Page() {
  return (
    <Providers>
      <ViewerApp />
    </Providers>
  );
}
