import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import RootError from "./error";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("root error boundary", () => {
  it("logs the exception without disclosing its raw message in the page", async () => {
    const user = userEvent.setup();
    const reset = vi.fn();
    const error = Object.assign(new Error("private backend detail"), {
      digest: "digest-123",
    });
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);

    render(<RootError error={error} reset={reset} />);

    expect(screen.queryByText("private backend detail")).not.toBeInTheDocument();
    expect(screen.getByText("Reference: digest-123")).toBeInTheDocument();
    await waitFor(() => expect(consoleError).toHaveBeenCalledWith(error));

    await user.click(screen.getByRole("button", { name: "Try Again" }));
    expect(reset).toHaveBeenCalledTimes(1);
  });
});
