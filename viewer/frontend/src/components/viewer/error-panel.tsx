import { AlertCircle } from "lucide-react";

export function ErrorPanel({ title, message }: { title: string; message: string }) {
  return (
    <div
      role="alert"
      className="rounded-md border border-danger-line bg-danger-soft p-3 text-sm text-danger shadow-panel"
    >
      <div className="flex items-center gap-2 font-semibold">
        <AlertCircle className="h-4 w-4" aria-hidden />
        {title}
      </div>
      <div className="mt-1 text-[#7f1d23]">{message}</div>
    </div>
  );
}
