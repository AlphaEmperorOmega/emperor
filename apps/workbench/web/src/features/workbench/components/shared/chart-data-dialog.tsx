"use client";

import { useId, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, TableProperties, X } from "lucide-react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { cn } from "@/lib/utils";

const CHART_DATA_PAGE_SIZE = 100;

export type ChartDataColumn<Row> = {
  key: string;
  label: string;
  align?: "left" | "right";
  render: (row: Row) => ReactNode;
};

export type ChartDataCompleteness = {
  incomplete: boolean;
  reason?: string | null;
  sourceRowCount?: number | null;
};

function returnedRowsLabel(count: number) {
  return `${count} returned ${count === 1 ? "row" : "rows"}`;
}

export function formatChartWallTime(wallTime: number) {
  const date = new Date(wallTime * 1000);
  return Number.isNaN(date.getTime()) ? String(wallTime) : date.toISOString();
}

export function ChartDataAction<Row>({
  chartTitle,
  columns,
  completeness,
  rowKey,
  rows,
}: {
  chartTitle: string;
  columns: readonly ChartDataColumn<Row>[];
  completeness?: ChartDataCompleteness;
  rowKey?: (row: Row, index: number) => string;
  rows: readonly Row[];
}) {
  const [isOpen, setIsOpen] = useState(false);
  const id = useId();

  return (
    <>
      <Button
        variant="ghost"
        className="h-touch shrink-0 gap-1.5 px-2 type-meta font-bold text-ink-dim hover:text-ink md:h-control-sm"
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        onClick={() => setIsOpen(true)}
      >
        <TableProperties className="h-3.5 w-3.5" aria-hidden />
        View chart data
      </Button>
      {isOpen && typeof document !== "undefined" &&
        createPortal(
          <ChartDataDialog
            id={id}
            chartTitle={chartTitle}
            columns={columns}
            completeness={completeness}
            rowKey={rowKey}
            rows={rows}
            onClose={() => setIsOpen(false)}
          />,
          document.body,
        )}
    </>
  );
}

function ChartDataDialog<Row>({
  id,
  chartTitle,
  columns,
  completeness,
  rowKey,
  rows,
  onClose,
}: {
  id: string;
  chartTitle: string;
  columns: readonly ChartDataColumn<Row>[];
  completeness?: ChartDataCompleteness;
  rowKey?: (row: Row, index: number) => string;
  rows: readonly Row[];
  onClose: () => void;
}) {
  const [pageIndex, setPageIndex] = useState(0);
  const titleId = `${id}-chart-data-title`;
  const descriptionId = `${id}-chart-data-description`;
  const pageCount = Math.max(1, Math.ceil(rows.length / CHART_DATA_PAGE_SIZE));
  const safePageIndex = Math.min(pageIndex, pageCount - 1);
  const startIndex = safePageIndex * CHART_DATA_PAGE_SIZE;
  const endIndex = Math.min(startIndex + CHART_DATA_PAGE_SIZE, rows.length);
  const visibleRows = rows.slice(startIndex, endIndex);
  const sourceCount = completeness?.sourceRowCount;
  const sourceCountCopy =
    typeof sourceCount === "number" && Number.isFinite(sourceCount)
      ? ` The source reported ${sourceCount} rows.`
      : "";

  return (
    <DialogShell
      labelledBy={titleId}
      describedBy={descriptionId}
      size="lg"
      panelVariant="surface"
      onClose={onClose}
      className="z-[90] bg-black/65 p-3 sm:p-4"
      panelClassName="min-h-0"
      header={
        <header className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <p className="type-label font-bold uppercase tracking-label text-ink-faint">
              Chart data
            </p>
            <h2 id={titleId} className="mt-0.5 truncate text-base font-semibold text-ink">
              {chartTitle}
            </h2>
            <p id={descriptionId} className="mt-1 text-sm text-ink-dim">
              Complete table of {returnedRowsLabel(rows.length)} represented by this chart.
            </p>
          </div>
          <IconButton
            label="Close chart data"
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
            onClick={onClose}
          />
        </header>
      }
      footer={
        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-line-soft px-4 py-3 sm:px-5">
          <span className="font-mono type-meta text-ink-dim" aria-live="polite">
            {rows.length === 0
              ? "No returned rows"
              : `Rows ${startIndex + 1}–${endIndex} of ${rows.length}`} · page{" "}
            {safePageIndex + 1} of {pageCount}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              className="h-touch px-2.5 type-compact md:h-control-sm"
              disabled={safePageIndex === 0}
              onClick={() => setPageIndex((current) => Math.max(0, current - 1))}
            >
              <ChevronLeft className="h-4 w-4" aria-hidden />
              Previous
            </Button>
            <Button
              variant="secondary"
              className="h-touch px-2.5 type-compact md:h-control-sm"
              disabled={safePageIndex >= pageCount - 1}
              onClick={() =>
                setPageIndex((current) => Math.min(pageCount - 1, current + 1))
              }
            >
              Next
              <ChevronRight className="h-4 w-4" aria-hidden />
            </Button>
          </div>
        </footer>
      }
    >
      <div className="min-h-0 flex-1 overflow-auto">
        {completeness?.incomplete && (
          <div
            role="note"
            className="border-b border-amber/30 bg-amber/[0.08] px-4 py-3 text-sm text-amber sm:px-5"
          >
            The API marked this dataset as incomplete. The table includes every
            returned row, but additional source data was omitted.{sourceCountCopy}{" "}
            {completeness.reason ?? "The response was truncated to stay within its payload limit."}
          </div>
        )}
        <table className="w-full min-w-max border-collapse text-left type-compact">
          <caption className="sr-only">
            {chartTitle}: {returnedRowsLabel(rows.length)}
          </caption>
          <thead className="sticky top-0 z-10 bg-panel-2 shadow-divider">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  className={cn(
                    "whitespace-nowrap border-b border-line px-4 py-2.5 type-label font-bold uppercase tracking-label text-ink-faint sm:px-5",
                    column.align === "right" && "text-right",
                  )}
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono text-ink-dim">
            {visibleRows.map((row, pageRowIndex) => {
              const absoluteIndex = startIndex + pageRowIndex;
              return (
                <tr
                  key={rowKey?.(row, absoluteIndex) ?? String(absoluteIndex)}
                  className="border-b border-line-soft odd:bg-white/[0.012] hover:bg-control-hover"
                >
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className={cn(
                        "whitespace-nowrap px-4 py-2 sm:px-5",
                        column.align === "right" && "text-right tabular-nums",
                      )}
                    >
                      {column.render(row)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 && (
          <p className="px-4 py-8 text-center text-sm text-ink-faint sm:px-5">
            The API returned no rows for this chart.
          </p>
        )}
      </div>
    </DialogShell>
  );
}
