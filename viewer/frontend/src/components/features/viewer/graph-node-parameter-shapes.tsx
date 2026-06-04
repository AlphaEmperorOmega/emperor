type ParameterShapeEntry = {
  key: string;
  label: string;
  shape: string;
};

export function GraphNodeParameterShapes({
  nodeId,
  entries,
}: {
  nodeId: string;
  entries: ParameterShapeEntry[];
}) {
  return (
    <div className="mt-2 grid shrink-0 gap-1">
      <div
        className="grid grid-cols-2 gap-1"
        data-testid={`parameter-shapes-${nodeId}`}
      >
        {entries.map((entry) => (
          <div
            key={entry.key}
            aria-label={`${entry.label} shape ${entry.shape}`}
            title={`${entry.label} shape: ${entry.shape}`}
            className="grid h-7 grid-cols-[18px_minmax(0,1fr)] items-center gap-1.5 rounded-[7px] border border-violet/25 bg-violet/15 px-2 text-[12px] leading-none shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]"
          >
            <span className="truncate font-semibold uppercase text-[#cdbcff]">
              {entry.label}
            </span>
            <span className="truncate font-mono text-ink">{entry.shape}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
