import { Loader2 } from "lucide-react";
import { type ReactNode } from "react";

type FullPageStatusProps = {
  title: string;
  detail?: string;
  icon: ReactNode;
  action?: ReactNode;
};

export function FullPageStatus({ title, detail, icon, action }: FullPageStatusProps) {
  return (
    <main className="grid min-h-screen place-items-center bg-bg p-6 text-ink">
      <div className="edge grid max-w-md justify-items-center gap-3 rounded-card p-6 text-center">
        <div className="flex h-10 w-10 items-center justify-center rounded-[10px] border border-line bg-white/[0.04] text-violet">
          {icon}
        </div>
        <div>
          <h1 className="text-lg font-bold">{title}</h1>
          {detail && <p className="mt-1 text-sm leading-6 text-ink-dim">{detail}</p>}
        </div>
        {action}
      </div>
    </main>
  );
}

export function FullPageLoading() {
  return (
    <FullPageStatus
      title="Loading viewer"
      icon={<Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
    />
  );
}
