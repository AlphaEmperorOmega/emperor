import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  installFetchMock,
  locationInspectResponse,
  neuronModelsResponse,
  renderViewer,
  resetViewerAppTestState,
} from "./viewer-app/support";

describe("NeuronCluster 3D popup", () => {
  beforeEach(resetViewerAppTestState);

  it("shows the 3D button when the neuron model type is selected and opens DialogShell", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByTestId("flow")).toBeInTheDocument();

    const panel = await screen.findByTestId("flow-panel-bottom-right");
    await user.click(
      within(panel).getByRole("button", { name: /open 3d cluster view/i }),
    );

    expect(
      await screen.findByRole("dialog", {
        name: /^3d neuron cluster model\.cluster$/i,
      }),
    ).toBeInTheDocument();
  });

  it("closes from the close button and Escape", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );
    await user.click(screen.getByRole("button", { name: /close 3d cluster view/i }));
    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", {
          name: /^3d neuron cluster model\.cluster$/i,
        }),
      ).not.toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /open 3d cluster view/i }));
    expect(
      await screen.findByRole("dialog", {
        name: /^3d neuron cluster model\.cluster$/i,
      }),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", {
          name: /^3d neuron cluster model\.cluster$/i,
        }),
      ).not.toBeInTheDocument();
    });
  });

  it("passes slice state to the scene and reveals mapped coordinates in full mode", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );
    expect(await screen.findByTestId("mock-cluster-3d-scene"))
      .toHaveAttribute("data-visible-x", "2");

    await user.click(screen.getByRole("button", { name: /hide x slice 2/i }));
    expect(await screen.findByTestId("mock-cluster-3d-scene"))
      .toHaveAttribute("data-visible-x", "1");

    await user.click(
      screen.getByRole("button", {
        name: /3d coordinate \(1, 1, 1\) initial/i,
      }),
    );

    expect(screen.getByRole("tab", { name: /full/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(await screen.findByTestId("node-model.cluster.neuron_1_1_1"))
      .toBeInTheDocument();
  });
});
