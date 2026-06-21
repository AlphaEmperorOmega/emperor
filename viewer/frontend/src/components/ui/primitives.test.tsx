import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { EdgeCard } from "@/components/ui/edge-card";
import { IconButton } from "@/components/ui/icon-button";
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

  it.each([
    ["success", "border-ok/30", "bg-ok/10", "text-ok"],
    ["warning", "border-amber/40", "bg-amber/[0.12]", "text-amber"],
    ["danger", "border-danger-line", "bg-danger-soft", "text-danger-text"],
    ["info", "border-blue/30", "bg-blue/10", "text-blue"],
    ["violet", "border-violet/30", "bg-violet/15", "text-violet"],
  ] as const)("applies the %s variant styling", (variant, borderClass, bgClass, textClass) => {
    render(<Badge variant={variant}>{variant}</Badge>);
    expect(screen.getByText(variant)).toHaveClass(borderClass, bgClass, textClass);
  });
});

describe("Button", () => {
  it("defaults to type=button and the secondary variant", () => {
    render(<Button>go</Button>);
    const button = screen.getByRole("button", { name: "go" });
    expect(button).toHaveAttribute("type", "button");
    expect(button.className).toContain("bg-control");
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

describe("IconButton", () => {
  it("uses label as the accessible name, defaults to type=button, and renders only the icon", () => {
    render(
      <IconButton
        label="Refresh preview"
        icon={<svg data-testid="refresh-icon" aria-hidden />}
      />,
    );
    const button = screen.getByRole("button", { name: "Refresh preview" });
    expect(button).toHaveAttribute("type", "button");
    expect(screen.getByTestId("refresh-icon")).toBeInTheDocument();
    expect(button).not.toHaveTextContent("Refresh preview");
  });

  it("fires clicks and respects disabled state", () => {
    const onClick = vi.fn();
    const { rerender } = render(
      <IconButton
        label="Delete run"
        icon={<span aria-hidden />}
        onClick={onClick}
      />,
    );
    const button = screen.getByRole("button", { name: "Delete run" });
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);

    rerender(
      <IconButton
        label="Delete run"
        icon={<span aria-hidden />}
        onClick={onClick}
        disabled
      />,
    );
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("passes through className after base, size, and variant classes", () => {
    render(
      <IconButton
        label="Custom action"
        icon={<span aria-hidden />}
        size="sm"
        variant="edge"
        className="custom-class bg-ok/10"
      />,
    );
    const button = screen.getByRole("button", { name: "Custom action" });
    expect(button).toHaveClass(
      "inline-flex",
      "h-8",
      "w-8",
      "rounded-control-md",
      "border-line",
      "text-ink-faint",
      "custom-class",
      "bg-ok/10",
    );
    expect(button).not.toHaveClass("bg-control");
  });

  it("applies md, ghost, and danger classes", () => {
    const { rerender } = render(
      <IconButton
        label="Ghost action"
        icon={<span aria-hidden />}
        size="md"
        variant="ghost"
      />,
    );
    const button = screen.getByRole("button", { name: "Ghost action" });
    expect(button).toHaveClass(
      "h-9",
      "w-9",
      "rounded-control",
      "border-transparent",
      "hover:bg-control-active",
    );

    rerender(
      <IconButton
        label="Danger action"
        icon={<span aria-hidden />}
        variant="danger"
      />,
    );
    expect(screen.getByRole("button", { name: "Danger action" })).toHaveClass(
      "border-transparent",
      "hover:border-danger-line",
      "hover:bg-danger-soft",
      "hover:text-danger-text",
    );
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
  it("renders a labelled radiogroup containing its segment buttons by default", () => {
    render(
      <SegmentedControl aria-label="Graph detail">
        <button type="button" role="radio" aria-checked="true">
          Simple
        </button>
      </SegmentedControl>,
    );
    const group = screen.getByRole("radiogroup", { name: "Graph detail" });
    expect(group).toContainElement(screen.getByRole("radio", { name: "Simple" }));
  });

  it("can render tablist semantics for real tab panels", () => {
    render(
      <SegmentedControl aria-label="Training config selector" variant="tablist">
        <button type="button" role="tab">
          Presets
        </button>
      </SegmentedControl>,
    );
    const tablist = screen.getByRole("tablist", {
      name: "Training config selector",
    });
    expect(tablist).toContainElement(screen.getByRole("tab", { name: "Presets" }));
  });
});

describe("StatusDot", () => {
  it("uses the online style when online and the danger style when offline", () => {
    const { container, rerender } = render(<StatusDot online />);
    expect(container.firstChild).toHaveClass("bg-ok");

    rerender(<StatusDot online={false} />);
    expect(container.firstChild).toHaveClass("bg-danger");
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
