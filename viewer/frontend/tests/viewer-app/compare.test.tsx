import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  installFetchMock,
  renderViewer,
  resetViewerAppTestState,
} from "./support";

describe("ViewerApp Compare Workspace", () => {
  beforeEach(resetViewerAppTestState);

  it("keeps compare content in a constrained scrollport after main menu navigation", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^compare$/i }));

    const heading = await screen.findByRole("heading", {
      name: /model comparison/i,
    });
    const scrollRoot = heading.closest(".overflow-y-auto");

    expect(scrollRoot).toBeInstanceOf(HTMLElement);
    expect(scrollRoot).toHaveClass("h-full", "min-h-0", "overflow-y-auto");
    expect(scrollRoot?.parentElement).toHaveClass("h-full", "overflow-hidden");
  });
});
