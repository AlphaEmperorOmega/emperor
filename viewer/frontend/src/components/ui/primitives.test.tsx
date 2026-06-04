import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { EdgeCard } from "@/components/ui/edge-card";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Select } from "@/components/ui/select";
import { StatusDot } from "@/components/ui/status-dot";
import { Switch } from "@/components/ui/switch";

// Render + role/behaviour smoke tests for the shared primitives, pinning the
// current public contract before they are reused more widely in Phase 1.

describe("Badge", () => {
  it("renders content in a span and merges className", () => {
    render(<Badge className="custom-class">label</Badge>);
    const badge = screen.getByText("label");
    expect(badge.tagName).toBe("SPAN");
    expect(badge).toHaveClass("custom-class");
  });

  it("applies the override variant styling", () => {
    render(<Badge variant="override">2 set</Badge>);
    expect(screen.getByText("2 set")).toHaveClass("text-violet");
  });

  it("applies the preset variant styling", () => {
    render(<Badge variant="preset">preset</Badge>);
    expect(screen.getByText("preset")).toHaveClass("text-amber");
  });
});

describe("Button", () => {
  it("defaults to type=button and the secondary variant", () => {
    render(<Button>go</Button>);
    const button = screen.getByRole("button", { name: "go" });
    expect(button).toHaveAttribute("type", "button");
    expect(button.className).toContain("bg-white/[0.035]");
  });

  it("applies the primary variant and fires onClick", () => {
    const onClick = vi.fn();
    render(
      <Button variant="primary" onClick={onClick}>
        run
      </Button>,
    );
    const button = screen.getByRole("button", { name: "run" });
    expect(button.className).toContain("bg-grad");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("respects the disabled attribute", () => {
    render(<Button disabled>nope</Button>);
    expect(screen.getByRole("button", { name: "nope" })).toBeDisabled();
  });
});

describe("Input", () => {
  it("renders a textbox and forwards props", () => {
    render(<Input placeholder="search" defaultValue="x" />);
    const input = screen.getByPlaceholderText("search");
    expect(input).toHaveValue("x");
  });
});

describe("Select", () => {
  it("renders a combobox with its options and fires onChange", () => {
    const onChange = vi.fn();
    render(
      <Select value="a" onChange={onChange}>
        <option value="a">A</option>
        <option value="b">B</option>
      </Select>,
    );
    const select = screen.getByRole("combobox");
    expect(select).toHaveValue("a");
    fireEvent.change(select, { target: { value: "b" } });
    expect(onChange).toHaveBeenCalled();
  });
});

describe("Switch", () => {
  it("exposes role=switch with aria-checked and toggles via onCheckedChange", () => {
    const onCheckedChange = vi.fn();
    render(<Switch checked={false} onCheckedChange={onCheckedChange} />);
    const toggle = screen.getByRole("switch");
    expect(toggle).toHaveAttribute("aria-checked", "false");
    fireEvent.click(toggle);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });
});

describe("Checkbox", () => {
  it("exposes a role=checkbox reflecting checked and toggles via onCheckedChange", () => {
    const onCheckedChange = vi.fn();
    render(
      <Checkbox checked={false} onCheckedChange={onCheckedChange} aria-label="pick" />,
    );
    const box = screen.getByRole("checkbox", { name: "pick" });
    expect(box).not.toBeChecked();
    fireEvent.click(box);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it("reflects the checked state on the native input", () => {
    render(
      <Checkbox checked onCheckedChange={() => {}} aria-label="on" />,
    );
    expect(screen.getByRole("checkbox", { name: "on" })).toBeChecked();
  });
});

describe("SegmentedControl", () => {
  it("renders a labelled tablist containing its segment buttons", () => {
    render(
      <SegmentedControl aria-label="Graph detail">
        <button type="button" role="tab">
          Simple
        </button>
      </SegmentedControl>,
    );
    const tablist = screen.getByRole("tablist", { name: "Graph detail" });
    expect(tablist).toContainElement(screen.getByRole("tab", { name: "Simple" }));
  });
});

describe("StatusDot", () => {
  it("uses the online style when online and the danger style when offline", () => {
    const { container, rerender } = render(<StatusDot online />);
    expect(container.firstChild).toHaveClass("bg-ok");

    rerender(<StatusDot online={false} />);
    expect(container.firstChild).toHaveClass("bg-[#fb7185]");
  });
});

describe("EdgeCard", () => {
  it("applies edge classes and the selected modifier", () => {
    const { container } = render(<EdgeCard selected>body</EdgeCard>);
    const card = container.firstChild as HTMLElement;
    expect(card).toHaveClass("edge");
    expect(card).toHaveClass("edge-sel");
    expect(card).toHaveTextContent("body");
  });
});
