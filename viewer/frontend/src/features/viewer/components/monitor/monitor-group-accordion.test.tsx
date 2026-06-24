import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MonitorGroupAccordion } from "@/features/viewer/components/monitor/monitor-group-accordion";
import { monitorGroupPanelId } from "@/lib/monitor/grouping";

describe("MonitorGroupAccordion", () => {
  it("preserves accordion semantics and child visibility", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    const panelId = monitorGroupPanelId("monitor-test", "Activations");

    function Harness() {
      const [isOpen, setIsOpen] = useState(true);

      return (
        <MonitorGroupAccordion
          idPrefix="monitor-test"
          group="Activations"
          count={2}
          countUnit="chart"
          isOpen={isOpen}
          onToggle={() => {
            onToggle();
            setIsOpen((current) => !current);
          }}
        >
          <div>Activation chart body</div>
        </MonitorGroupAccordion>
      );
    }

    render(<Harness />);

    const trigger = screen.getByRole("button", { name: /activations/i });
    const surface = trigger.closest("section");
    if (!(surface instanceof HTMLElement)) {
      throw new Error("Expected monitor accordion to render inside a section");
    }
    expect(surface).toHaveClass(
      "rounded-[10px]",
      "border",
      "border-line",
      "bg-white/[0.018]",
      "p-0",
    );
    expect(surface).not.toHaveClass("edge", "rounded-card");
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(trigger).toHaveAttribute("aria-controls", panelId);
    expect(screen.getByText("Activation chart body")).toBeInTheDocument();

    await user.click(trigger);

    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveAttribute("aria-controls", panelId);
    expect(screen.queryByText("Activation chart body")).not.toBeInTheDocument();
  });
});
