import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  commandForShell,
  TrainingShellSelector,
  useTrainingShell,
} from "@/features/workbench/components/training/training-shell-selector";
import {
  getTrainingShellServerSnapshot,
  getTrainingShellSnapshot,
  setTrainingShell,
  TRAINING_SHELL_CHANGE_EVENT,
  TRAINING_SHELL_STORAGE_KEY,
} from "@/features/workbench/components/training/training-shell-store";

const originalPlatform = Object.getOwnPropertyDescriptor(navigator, "platform");

function ShellFixture() {
  const { shell, setShell } = useTrainingShell();
  return <TrainingShellSelector shell={shell} onChange={setShell} />;
}

afterEach(() => {
  vi.restoreAllMocks();
  act(() => setTrainingShell("posix"));
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

  it("uses the requested canonical shell projection", () => {
    expect(
      commandForShell(
        {
          commands: {
            posix: "mise run experiment -- --posix",
            powershell: "mise run experiment -- --powershell",
          },
        },
        "powershell",
      ),
    ).toBe("mise run experiment -- --powershell");
  });

  it("uses a deterministic POSIX server snapshot", () => {
    expect(getTrainingShellServerSnapshot()).toBe("posix");
  });

  it("observes storage and custom events while keeping focus-independent state", async () => {
    const rendered = render(<ShellFixture />);

    window.localStorage.setItem(TRAINING_SHELL_STORAGE_KEY, "powershell");
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: TRAINING_SHELL_STORAGE_KEY,
          newValue: "powershell",
          storageArea: null,
        }),
      );
    });
    await waitFor(() => {
      expect(screen.getByRole("radio", { name: "PowerShell" })).toBeChecked();
    });

    window.localStorage.setItem(TRAINING_SHELL_STORAGE_KEY, "posix");
    act(() => {
      window.dispatchEvent(new Event(TRAINING_SHELL_CHANGE_EVENT));
    });
    await waitFor(() => {
      expect(screen.getByRole("radio", { name: "POSIX" })).toBeChecked();
    });

    rendered.unmount();
  });

  it("publishes setter events and keeps the in-memory shell when storage fails", () => {
    const eventListener = vi.fn();
    window.addEventListener(TRAINING_SHELL_CHANGE_EVENT, eventListener);
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("write unavailable");
      });

    act(() => setTrainingShell("powershell"));

    expect(getTrainingShellSnapshot()).toBe("powershell");
    expect(eventListener).toHaveBeenCalledTimes(1);
    expect(
      (eventListener.mock.calls[0][0] as CustomEvent<string>).detail,
    ).toBe("powershell");

    setItem.mockRestore();
    window.removeEventListener(TRAINING_SHELL_CHANGE_EVENT, eventListener);
  });
});
