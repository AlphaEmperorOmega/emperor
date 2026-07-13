import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import {
  commandForShell,
  TrainingShellSelector,
  useTrainingShell,
} from "@/features/workbench/components/training/training-shell-selector";

const originalPlatform = Object.getOwnPropertyDescriptor(navigator, "platform");

function ShellFixture() {
  const { shell, setShell } = useTrainingShell();
  return <TrainingShellSelector shell={shell} onChange={setShell} />;
}

afterEach(() => {
  window.localStorage.clear();
  if (originalPlatform) {
    Object.defineProperty(navigator, "platform", originalPlatform);
  }
});

describe("TrainingShellSelector", () => {
  it("suggests PowerShell on Windows and permits a persistent POSIX override", async () => {
    Object.defineProperty(navigator, "platform", {
      configurable: true,
      value: "Win32",
    });
    const user = userEvent.setup();
    const first = render(<ShellFixture />);

    await waitFor(() => {
      expect(screen.getByRole("radio", { name: "PowerShell" })).toBeChecked();
    });
    await user.click(screen.getByRole("radio", { name: "POSIX" }));
    expect(screen.getByRole("radio", { name: "POSIX" })).toBeChecked();

    first.unmount();
    render(<ShellFixture />);
    await waitFor(() => {
      expect(screen.getByRole("radio", { name: "POSIX" })).toBeChecked();
    });
  });

  it("uses the compatibility command when a shell projection is absent", () => {
    expect(commandForShell({ command: "source experiment.sh" }, "powershell"))
      .toBe("source experiment.sh");
  });
});
