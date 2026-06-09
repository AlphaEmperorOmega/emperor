import { Badge } from "@/components/ui/badge";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import {
  compareHeader,
  type CompareEntryData,
  type ConfigDiffRow,
  type StatRow,
} from "./derive";

export function ComparisonTable({
  title,
  rows,
  entries,
  emptyMessage,
}: {
  title: string;
  rows: StatRow[];
  entries: CompareEntryData[];
  emptyMessage: string;
}) {
  if (rows.length === 0) {
    return (
      <InlineStatus compact>
        {emptyMessage}
      </InlineStatus>
    );
  }

  return (
    <section className="overflow-hidden rounded-[12px] border border-line-soft bg-black/16">
      <div className="flex items-center justify-between gap-3 border-b border-line-soft px-3 py-2">
        <h3 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          {title}
        </h3>
        <Badge>{rows.length}</Badge>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-line-soft text-xs uppercase tracking-[0.08em] text-ink-faint">
              <th className="w-[220px] px-3 py-2 font-bold">Metric</th>
              {entries.map((entry, index) => (
                <th key={entry.entry.id} className="px-3 py-2 font-bold">
                  {compareHeader(entry, index)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.label}
                className="border-b border-line-soft last:border-b-0"
              >
                <th className="px-3 py-2 align-top text-xs font-semibold text-ink-dim">
                  {row.label}
                </th>
                {row.values.map((value, index) => (
                  <td
                    key={`${row.label}-${entries[index]?.entry.id ?? index}`}
                    className={
                      row.changed
                        ? "px-3 py-2 align-top font-mono text-xs text-amber"
                        : "px-3 py-2 align-top font-mono text-xs text-ink-dim"
                    }
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function ConfigDiffTable({
  rows,
  entries,
}: {
  rows: ConfigDiffRow[];
  entries: CompareEntryData[];
}) {
  if (rows.length === 0) {
    return (
      <InlineStatus compact>
        No changed config values for the selected targets.
      </InlineStatus>
    );
  }

  return (
    <section className="overflow-hidden rounded-[12px] border border-line-soft bg-black/16">
      <div className="flex items-center justify-between gap-3 border-b border-line-soft px-3 py-2">
        <h3 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          Changed Config Values
        </h3>
        <Badge>{rows.length}</Badge>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-line-soft text-xs uppercase tracking-[0.08em] text-ink-faint">
              <th className="w-[240px] px-3 py-2 font-bold">Field</th>
              {entries.map((entry, index) => (
                <th key={entry.entry.id} className="px-3 py-2 font-bold">
                  {compareHeader(entry, index)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className="border-b border-line-soft last:border-b-0">
                <th className="px-3 py-2 align-top">
                  <span className="block text-xs font-semibold text-ink">
                    {row.label}
                  </span>
                  <span className="block font-mono text-[11px] text-ink-faint">
                    {row.section}
                  </span>
                </th>
                {row.values.map((value, index) => (
                  <td
                    key={`${row.key}-${entries[index]?.entry.id ?? index}`}
                    className="px-3 py-2 align-top font-mono text-xs text-amber"
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
