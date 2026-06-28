import dynamic from "next/dynamic";
import { type ComponentProps } from "react";
import { type MonitorChartsModal } from "@/features/viewer/components/monitor/monitor-charts-modal";

export type LazyMonitorChartsModalProps = ComponentProps<typeof MonitorChartsModal>;

export const LazyMonitorChartsModal = dynamic<LazyMonitorChartsModalProps>(
  () =>
    import("@/features/viewer/components/monitor/monitor-charts-modal").then(
      (module) => module.MonitorChartsModal,
    ),
  { ssr: false },
);
