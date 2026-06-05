import { Badge } from "@/components/ui/badge";
import { formatCompactCount, formatExactCount } from "@/lib/graph";

const SIMPLE_PARAMETER_BADGE_CLASS =
  "shrink-0 whitespace-nowrap border-violet/25 bg-violet/15 px-1.5 py-0.5 font-mono text-[11px] leading-none text-[#cdbcff] [overflow-wrap:normal]";
const SIMPLE_DIMS_BADGE_CLASS =
  "shrink-0 whitespace-nowrap border-line bg-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] leading-none text-ink-dim [overflow-wrap:normal]";
const PARAMETER_BADGE_CLASS =
  "h-6 shrink-0 items-center whitespace-nowrap border-violet/25 bg-violet/15 px-2 py-0 font-mono text-xs leading-none text-[#cdbcff] [overflow-wrap:normal]";
const CHILD_BADGE_CLASS =
  "h-6 shrink-0 items-center whitespace-nowrap border-line bg-white/[0.04] px-2 py-0 font-sans text-xs font-medium leading-none text-ink-dim [overflow-wrap:normal]";

function GraphNodeParameterBadge({
  parameterCount,
  text,
  className,
}: {
  parameterCount: number;
  text: string;
  className: string;
}) {
  return (
    <Badge
      className={className}
      title={`${formatExactCount(parameterCount)} parameters`}
    >
      {text}
    </Badge>
  );
}

function GraphNodeChildBadge({ childCount }: { childCount: number }) {
  return (
    <Badge className={CHILD_BADGE_CLASS}>
      {childCount} {childCount === 1 ? "child" : "children"}
    </Badge>
  );
}

export function GraphNodeSimpleBadges({
  parameterCount,
  parameterText,
  dimsText,
}: {
  parameterCount: number;
  parameterText?: string;
  dimsText?: string;
}) {
  return (
    <>
      {parameterText && (
        <GraphNodeParameterBadge
          parameterCount={parameterCount}
          text={parameterText}
          className={SIMPLE_PARAMETER_BADGE_CLASS}
        />
      )}
      {dimsText && (
        <Badge
          className={SIMPLE_DIMS_BADGE_CLASS}
          title={`input/output: ${dimsText}`}
        >
          {dimsText}
        </Badge>
      )}
    </>
  );
}

export function GraphNodeInlineBadges({
  parameterCount,
  childCount,
}: {
  parameterCount: number;
  childCount: number;
}) {
  return (
    <>
      {parameterCount > 0 && (
        <GraphNodeParameterBadge
          parameterCount={parameterCount}
          text={formatCompactCount(parameterCount)}
          className={PARAMETER_BADGE_CLASS}
        />
      )}
      {childCount > 0 && <GraphNodeChildBadge childCount={childCount} />}
    </>
  );
}

export function GraphNodeBadgeRow({
  nodeId,
  parameterCount,
  childCount,
}: {
  nodeId: string;
  parameterCount: number;
  childCount: number;
}) {
  return (
    <div
      className="mt-1 flex h-6 min-w-0 items-center gap-1.5 overflow-hidden"
      data-testid={`graph-node-badges-${nodeId}`}
    >
      <GraphNodeInlineBadges
        parameterCount={parameterCount}
        childCount={childCount}
      />
    </div>
  );
}
