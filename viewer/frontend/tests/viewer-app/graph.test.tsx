import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  installFetchMock,
  inspectResponse,
  locationInspectResponse,
  manyRepeatedLayersInspectResponse,
  mechanismChildrenInspectResponse,
  mechanismMetadataInspectResponse,
  neuronModelsResponse,
  openFullConfig,
  parameterShapeInspectResponse,
  renderViewer,
  repeatedLayersInspectResponse,
  resetViewerAppTestState,
  selectTargetOption,
  stackContainerInspectResponse,
  tallSummaryInspectResponse,
  typeConfigFieldValue,
} from "./support";

describe("ViewerApp Graph Workspace", () => {
  beforeEach(resetViewerAppTestState);

  it("renders graph nodes progressively by default", async () => {
    installFetchMock();
    renderViewer();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(screen.getByTestId("flow")).toHaveAttribute(
      "data-only-render-visible-elements",
      "true",
    );
    expect(screen.getByTestId("flow")).toHaveAttribute(
      "data-has-move-handlers",
      "true",
    );
    expect(screen.getByTestId("edge-model-main_model.0")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();
  });

  it("shows child count beside the graph card title", async () => {
    installFetchMock({ inspectResponse: repeatedLayersInspectResponse });
    renderViewer();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByText("3 children")).toBeInTheDocument();
  });

  it("renders total params in the right sidebar summary", async () => {
    installFetchMock();
    renderViewer();

    const heading = await screen.findByRole("heading", { name: /node details/i });
    const sidebar = heading.closest("aside");
    if (!sidebar) {
      throw new Error("Expected Node Details heading to render inside the sidebar");
    }

    const paramsValue = await within(sidebar).findByTitle("65,792 parameters");
    const paramsCard = paramsValue.parentElement;
    if (!paramsCard) {
      throw new Error("Expected total params value to render inside a summary card");
    }

    expect(within(paramsCard).getByText(/^Params$/)).toBeInTheDocument();
    expect(paramsValue).toHaveTextContent("65.8K");
    expect(within(sidebar).queryByText(/^Nodes$/)).not.toBeInTheDocument();
    expect(within(sidebar).queryByText(/^Edges$/)).not.toBeInTheDocument();
  });

  it("renders graph parameter badges only for nodes with parameters", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByTitle("263,168 bytes of parameter tensors"))
      .toHaveTextContent("0.25 MB");

    const layerNode = await screen.findByTestId("node-main_model.0");
    expect(within(layerNode).getByTitle("33,024 parameters")).toHaveTextContent("33K");
    expect(within(layerNode).queryByTitle(/bytes of parameter tensors/i))
      .not.toBeInTheDocument();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const gateNode = await screen.findByTestId("node-main_model.0.gate_model");
    expect(within(gateNode).queryByTitle(/parameters/)).not.toBeInTheDocument();
  });

  it("refreshes the root model size when the target preset changes", async () => {
    const largerPresetInspectResponse = {
      ...inspectResponse,
      preset: "recurrent-gating-halting",
      parameterCount: 524288,
      parameterSizeBytes: 2 * 1024 * 1024,
      nodes: inspectResponse.nodes.map((node) =>
        node.id === "model"
          ? {
              ...node,
              parameterCount: 524288,
              parameterSizeBytes: 2 * 1024 * 1024,
            }
          : node,
      ),
    };
    let inspectBodies: unknown[] = [];
    ({ inspectBodies } = installFetchMock({
      inspectResponseFactory: (requestIndex) => {
        const request = inspectBodies[requestIndex] as { preset?: string } | undefined;
        return request?.preset === "recurrent-gating-halting"
          ? largerPresetInspectResponse
          : inspectResponse;
      },
    }));
    renderViewer();
    const user = userEvent.setup();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByTitle("263,168 bytes of parameter tensors"))
      .toHaveTextContent("0.25 MB");
    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    const initialRequestCount = inspectBodies.length;

    await selectTargetOption(user, "preset", "recurrent-gating-halting");

    await waitFor(() =>
      expect(inspectBodies.length).toBeGreaterThan(initialRequestCount),
    );
    expect(inspectBodies.at(-1)).toEqual({
      modelType: "linears",
      model: "linear",
      preset: "recurrent-gating-halting",
      dataset: "Mnist",
      overrides: { stack_hidden_dim: "128" },
    });
    const refreshedModelNode = await screen.findByTestId("node-model");
    expect(within(refreshedModelNode).getByTitle("2,097,152 bytes of parameter tensors"))
      .toHaveTextContent("2 MB");
  });

  it("renders layer dims on the inner-model graph-card row", async () => {
    installFetchMock();
    renderViewer();

    const layerNode = await screen.findByTestId("node-main_model.0");
    const matchingSummaries = within(layerNode).getAllByLabelText("LinearLayer 128 -> 128");
    const summary = matchingSummaries[0];
    expect(matchingSummaries).toHaveLength(2);
    expect(summary).toHaveAttribute("title", "LinearLayer 128 -> 128");
    expect(within(summary).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(summary).getByText("128 -> 128")).toBeInTheDocument();
    expect(within(layerNode).queryByText("shapeTransition")).not.toBeInTheDocument();
  });

  it("renders weight and bias shapes only on the graph cards that own them", async () => {
    installFetchMock({ inspectResponse: parameterShapeInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    const parentNode = await screen.findByTestId("node-main_model.0");
    expect(within(parentNode).queryByTestId("parameter-shapes-main_model.0")).not.toBeInTheDocument();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const ownerNode = await screen.findByTestId("node-main_model.0.model");
    const shapes = within(ownerNode).getByTestId("parameter-shapes-main_model.0.model");
    expect(within(shapes).getByLabelText("W shape 128 x 128")).toBeInTheDocument();
    expect(within(shapes).getByLabelText("b shape 128")).toBeInTheDocument();
    expect(within(screen.getByTestId("node-main_model.0")).queryByText("128 x 128")).not.toBeInTheDocument();
  });

  it("renders repeated layer children as separate graph card pills", async () => {
    installFetchMock({ inspectResponse: repeatedLayersInspectResponse });
    renderViewer();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByLabelText("Layer 0 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).getByLabelText("Layer 1 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).getByLabelText("Layer 2 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByText("Layer -> LinearLayer")).not.toBeInTheDocument();
  });

  it("summarizes long layer stacks with an ellipsis and total count", async () => {
    installFetchMock({ inspectResponse: manyRepeatedLayersInspectResponse });
    renderViewer();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByLabelText("Layer 0 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).getByLabelText("Layer 1 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByLabelText("Layer 2 LinearLayer")).not.toBeInTheDocument();
    expect(within(modelNode).getByTitle("6 more layers")).toHaveTextContent("...");
    expect(within(modelNode).getByLabelText("Layer 8 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByText("9 layers")).not.toBeInTheDocument();
  });

  it("builds basic-mode child summaries from the filtered graph", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    const summaries = within(layerNode).getByTestId("child-summaries-main_model.0");
    expect(within(summaries).getAllByText("LinearLayer")).toHaveLength(2);
    expect(within(layerNode).getByText("Gate")).toBeInTheDocument();
    expect(within(layerNode).queryByText("Dropout")).not.toBeInTheDocument();
    expect(within(layerNode).queryByText("LayerNorm")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /full/i }));

    const fullLayerNode = screen.getByTestId("node-main_model.0");
    expect(within(fullLayerNode).getByText("Dropout")).toBeInTheDocument();
    expect(within(fullLayerNode).getByText("LayerNorm")).toBeInTheDocument();
    expect(within(fullLayerNode).getByText("SelfAttentionProcessor")).toBeInTheDocument();
  });

  it("renders simple graph cards with inline metrics only", async () => {
    installFetchMock({ inspectResponse: parameterShapeInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
    const expandedLayerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(expandedLayerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );
    expect(within(screen.getByTestId("node-main_model.0")).getByText("activation"))
      .toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /simple/i }));

    const layerNode = await screen.findByTestId("node-main_model.0");
    expect(within(layerNode).getByText("Layer")).toBeInTheDocument();
    expect(
      within(layerNode).getByRole("button", { name: /^collapse tree main_model\.0$/i }),
    ).toBeInTheDocument();
    expect(within(layerNode).queryByText("main_model.0")).not.toBeInTheDocument();
    expect(within(layerNode).getByTitle("33,024 parameters")).toHaveTextContent("33K params");
    expect(within(layerNode).getByTitle("input/output: 128 -> 128")).toHaveTextContent(
      "128 -> 128",
    );
    expect(within(layerNode).queryByText("3 children")).not.toBeInTheDocument();
    expect(within(layerNode).queryByTestId("child-summaries-main_model.0")).not.toBeInTheDocument();
    expect(within(layerNode).queryByTestId("parameter-shapes-main_model.0")).not.toBeInTheDocument();
    expect(
      within(layerNode).queryByRole("button", { name: /details for main_model\.0/i }),
    ).not.toBeInTheDocument();
    expect(within(layerNode).queryByText("activation")).not.toBeInTheDocument();
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /basic/i }));

    const basicLayerNode = screen.getByTestId("node-main_model.0");
    expect(within(basicLayerNode).getByRole("button", { name: /details for main_model\.0/i }))
      .toBeInTheDocument();
    expect(within(basicLayerNode).getByText("activation")).toBeInTheDocument();
    expect(within(basicLayerNode).getByTestId("child-summaries-main_model.0"))
      .toBeInTheDocument();
  });

  it("renders stack-derived dims on simple stack container cards", async () => {
    installFetchMock({ inspectResponse: stackContainerInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("radio", { name: /simple/i }));

    const stackNode = await screen.findByTestId("node-main_model");
    expect(within(stackNode).getByText("LayerStack")).toBeInTheDocument();
    expect(within(stackNode).getByTitle("65,792 parameters")).toHaveTextContent("65.8K params");
    expect(within(stackNode).getByTitle("input/output: 256 -> 10")).toHaveTextContent(
      "256 -> 10",
    );
    expect(within(stackNode).queryByTestId("stack-diagram-main_model")).not.toBeInTheDocument();
    expect(within(stackNode).queryByTestId("child-summaries-main_model")).not.toBeInTheDocument();
    expect(
      within(stackNode).queryByRole("button", { name: /details for main_model/i }),
    ).not.toBeInTheDocument();
  });

  it("expands simple-mode cards using the basic graph topology", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("radio", { name: /simple/i }));

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /^expand tree main_model\.0$/i }),
    );

    const modelNode = await screen.findByTestId("node-main_model.0.model");
    const projectedNode = await screen.findByTestId("node-main_model.0.processor.projection");
    expect(within(modelNode).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByText("main_model.0.model")).not.toBeInTheDocument();
    expect(within(modelNode).getByTitle("16,512 parameters")).toHaveTextContent("16.5K params");
    expect(within(modelNode).getByTitle("input/output: 128 -> 128")).toHaveTextContent(
      "128 -> 128",
    );
    expect(within(projectedNode).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(projectedNode).getByTitle("16,512 parameters")).toHaveTextContent(
      "16.5K params",
    );
    expect(screen.queryByTestId("node-main_model.0.processor")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.dropout_module")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.layer_norm_module")).not.toBeInTheDocument();
  });

  it("renders enabled gate and halting metadata as child summary rows", async () => {
    installFetchMock({ inspectResponse: mechanismMetadataInspectResponse });
    renderViewer();

    const controllerNode = await screen.findByTestId("node-controller");
    expect(within(controllerNode).getByText("Gate")).toBeInTheDocument();
    expect(within(controllerNode).getByText("Halting mechanism")).toBeInTheDocument();
  });

  it("does not duplicate mechanism summaries when matching child nodes exist", async () => {
    installFetchMock({ inspectResponse: mechanismChildrenInspectResponse });
    renderViewer();

    const controllerNode = await screen.findByTestId("node-controller");
    const summaries = within(controllerNode).getByTestId("child-summaries-controller");
    expect(within(summaries).getAllByText("Gate")).toHaveLength(1);
    expect(within(summaries).getAllByText("Halting mechanism")).toHaveLength(1);
  });

  it("uses child summary rows when computing graph card height", async () => {
    installFetchMock({ inspectResponse: tallSummaryInspectResponse });
    renderViewer();

    const blockNode = await screen.findByTestId("node-block");
    expect(Number(blockNode.getAttribute("data-height"))).toBeGreaterThan(132);
    expect(within(blockNode).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(blockNode).getByText("AttentionLayer")).toBeInTheDocument();
    expect(within(blockNode).getByText("Embedding")).toBeInTheDocument();
    expect(within(blockNode).getByText("OutputHead")).toBeInTheDocument();
  });

  it("expands graph nodes and collapses the opened graph", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerCard = await screen.findByRole("button", {
      name: /select and expand main_model\.0/i,
    });
    await user.click(layerCard);
    expect(layerCard).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(screen.getByTestId("edge-main_model.0-main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByText("Sequential")).toBeInTheDocument();
    expect(screen.getByText("Gate Model · main_model.0.gate_model")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.processor.projection")).toBeInTheDocument();
    expect(
      screen.getByTestId("edge-main_model.0-main_model.0.processor.projection"),
    ).toBeInTheDocument();
    expect(screen.queryByText("SelfAttentionProcessor")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse all/i }));
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
    expect(screen.queryByText("main_model.0.processor.projection")).not.toBeInTheDocument();
  });

  it("opens the structure tree and reveals a clicked graph path", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByTestId("node-main_model.0")).toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.model")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^open graph structure$/i }));

    let structurePanel = await screen.findByTestId("graph-structure-panel");
    expect(
      within(structurePanel).getByRole("button", {
        name: /^reveal main_model\.0 in graph$/i,
      }),
    ).toBeInTheDocument();

    await user.click(
      within(structurePanel).getByRole("button", {
        name: /^expand structure main_model\.0$/i,
      }),
    );
    await user.click(
      within(structurePanel).getByRole("button", {
        name: /^reveal main_model\.0\.model in graph$/i,
      }),
    );

    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    const detailsHeading = await screen.findByRole("heading", { name: /node details/i });
    const detailsPanel = detailsHeading.closest("aside");
    if (!detailsPanel) {
      throw new Error("Expected Node Details heading to render inside the sidebar");
    }
    expect(within(detailsPanel).getAllByText("main_model.0.model").length).toBeGreaterThan(0);
    expect(within(detailsPanel).getByText("16,512")).toBeInTheDocument();
    structurePanel = await screen.findByTestId("graph-structure-panel");
    expect(
      within(structurePanel).getByRole("button", {
        name: /^reveal main_model\.0\.model in graph$/i,
      }),
    ).toHaveAttribute("aria-current", "true");
  });

  it("collapses an expanded graph card when its body is clicked again", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /select and collapse main_model\.0/i }),
    ).toHaveAttribute("aria-expanded", "true");

    await user.click(screen.getByRole("button", { name: /select and collapse main_model\.0/i }));

    expect(screen.queryByTestId("node-main_model.0.model")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select and expand main_model\.0/i }))
      .toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
  });

  it("keeps root graph cards expanded when their body is clicked", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const rootCard = await screen.findByRole("button", { name: /^select model$/i });
    expect(await screen.findByTestId("node-main_model.0")).toBeInTheDocument();

    await user.click(rootCard);

    expect(screen.getByTestId("node-main_model.0")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /select and collapse model/i }),
    ).not.toBeInTheDocument();
  });

  it("selects leaf graph cards without changing expansion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    await user.click(
      await screen.findByRole("button", { name: /^select main_model\.0\.model$/i }),
    );

    expect(screen.getByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select and collapse main_model\.0/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("16,512")).toBeInTheDocument();
  });

  it("expands another card on the first body click after one card is already open", async () => {
    installFetchMock({ inspectResponse: repeatedLayersInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /select and expand main_model\.1/i }));

    expect(await screen.findByTestId("node-main_model.1.model")).toBeInTheDocument();
  });

  it("expands and collapses the current-mode subtree from the chevron", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(screen.getByRole("radio", { name: /full/i }));
    expect(await screen.findByTestId("node-loss_fn")).toBeInTheDocument();

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /^expand tree main_model\.0$/i }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.processor")).toBeInTheDocument();
    expect(
      await screen.findByTestId("node-main_model.0.processor.projection"),
    ).toBeInTheDocument();

    const expandedLayerNode = screen.getByTestId("node-main_model.0");
    await user.click(
      within(expandedLayerNode).getByRole("button", { name: /^collapse tree main_model\.0$/i }),
    );
    expect(screen.queryByTestId("node-main_model.0.model")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.processor")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("node-main_model.0.processor.projection"),
    ).not.toBeInTheDocument();

    const collapsedLayerNode = screen.getByTestId("node-main_model.0");
    await user.click(
      within(collapsedLayerNode).getByRole("button", {
        name: /select and expand main_model\.0/i,
      }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.processor")).toBeInTheDocument();
    expect(
      screen.queryByTestId("node-main_model.0.processor.projection"),
    ).not.toBeInTheDocument();
  });

  it("keeps chevron expansion in basic detail mode and does not select the card", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /^expand tree main_model\.0$/i }),
    );

    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(
      await screen.findByTestId("node-main_model.0.processor.projection"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.processor")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.dropout_module")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.layer_norm_module")).not.toBeInTheDocument();
    expect(screen.queryByText("BEFORE")).not.toBeInTheDocument();
  });

  it("keeps opened graph cards visibly separated", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const directChildY = Number(
      screen.getByTestId("node-main_model.0.model").getAttribute("data-y"),
    );
    const projectedChildY = Number(
      screen.getByTestId("node-main_model.0.processor.projection").getAttribute("data-y"),
    );

    expect(Math.abs(projectedChildY - directChildY)).toBeGreaterThanOrEqual(148);
  });

  it("opens card details without expanding the subgraph", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const layerNode = screen.getByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );

    expect(within(layerNode).queryByText("dims")).not.toBeInTheDocument();
    expect(within(layerNode).getByText("activation")).toBeInTheDocument();
    expect(within(layerNode).getByText("dropout")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("relayouts graph cards when details change card height", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const layerNode = screen.getByTestId("node-main_model.0");
    const initialHeight = Number(layerNode.getAttribute("data-height"));
    const initialY = Number(layerNode.getAttribute("data-y"));

    await user.click(
      within(layerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );

    const expandedLayerNode = screen.getByTestId("node-main_model.0");
    expect(Number(expandedLayerNode.getAttribute("data-height"))).toBeGreaterThan(initialHeight);
    expect(Number(expandedLayerNode.getAttribute("data-y"))).not.toBe(initialY);
  });

  it("collapse all keeps card detail accordions open", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse all/i }));

    const visibleLayerNode = screen.getByTestId("node-main_model.0");
    expect(within(visibleLayerNode).getByText("activation")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("can switch between opened and entire graph scopes", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /entire/i }));
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.processor.projection")).toBeInTheDocument();
    expect(
      screen.getByTestId("edge-main_model.0-main_model.0.processor.projection"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /opened/i }));
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("selects cards in entire scope without changing opened-scope expansion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /entire/i }));

    await user.click(await screen.findByRole("button", { name: /^select main_model\.0$/i }));
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /opened/i }));
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("switches between basic and full graph detail", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /full/i }));
    expect(await screen.findByTestId("node-main_model.0.dropout_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.layer_norm_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.processor")).toBeInTheDocument();
    expect(
      screen.queryByTestId("node-main_model.0.processor.projection"),
    ).not.toBeInTheDocument();
    expect(await screen.findByTestId("node-loss_fn")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /basic/i }));
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("SelfAttentionProcessor")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();
  });

  it("shows the complete graph only in full entire mode", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("radio", { name: /entire/i }));
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /full/i }));
    expect(await screen.findByTestId("node-main_model.0.dropout_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.layer_norm_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-loss_fn")).toBeInTheDocument();
    expect(await screen.findByTestId("node-metrics")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.processor.projection")).toBeInTheDocument();
  });

  it("collapse all keeps the selected graph detail mode", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await user.click(screen.getByRole("radio", { name: /full/i }));
    expect(await screen.findByTestId("node-main_model.0.dropout_module")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse all/i }));
    expect(screen.getByRole("radio", { name: /full/i })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(screen.queryByTestId("node-main_model.0.dropout_module")).not.toBeInTheDocument();
    expect(
      within(screen.getByTestId("node-main_model.0")).getByText("Dropout"),
    ).toBeInTheDocument();
  });

  it("renders graph-only preview controls with the selected cluster locations card", async () => {
    installFetchMock({ inspectResponse: locationInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByTestId("flow")).toBeInTheDocument();
    expect(
      screen.queryByRole("radiogroup", { name: /preview visualization/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("radio", { name: /^parameters$/i }),
    ).not.toBeInTheDocument();
    const graphDetailTabs = screen.getByRole("radiogroup", { name: /graph detail/i });
    const graphScopeTabs = screen.getByRole("radiogroup", { name: /graph scope/i });
    expect(
      screen.queryByRole("radiogroup", { name: /graph kind/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("radio", { name: /^operations$/i }),
    ).not.toBeInTheDocument();
    expect([
      ...within(graphDetailTabs).getAllByRole("radio"),
      ...within(graphScopeTabs).getAllByRole("radio"),
    ].map((tab) => tab.textContent)).toEqual([
      "Simple",
      "Basic",
      "Full",
      "Opened",
      "Entire",
    ]);
    expect(graphDetailTabs).toBeInTheDocument();
    expect(within(graphDetailTabs).getByRole("radio", { name: /basic/i }))
      .toHaveAttribute("aria-checked", "true");
    expect(graphScopeTabs).toBeInTheDocument();
    expect(within(graphScopeTabs).getByRole("radio", { name: /opened/i }))
      .toHaveAttribute("aria-checked", "true");
    expect(screen.queryByTestId("flow-panel-bottom-right")).not.toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: /^select model\.cluster$/i }));
    const panel = await screen.findByTestId("flow-panel-bottom-right");

    expect(screen.queryByTestId("model-structure-card")).not.toBeInTheDocument();
    expect(screen.queryByText("Model Structure")).not.toBeInTheDocument();
    expect(
      within(panel).queryByRole("button", { name: /open 3d cluster view/i }),
    ).not.toBeInTheDocument();
    let card = within(panel).getByTestId("graph-locations-card");
    expect(within(card).getByRole("button", { name: /show cluster locations/i }))
      .toBeInTheDocument();
    expect(panel).toHaveClass("nodrag", "nopan", "hidden", "xl:block");
    expect(panel).toHaveStyle({ right: "28px", bottom: "24px" });

    await user.click(within(card).getByRole("button", { name: /show cluster locations/i }));
    card = within(panel).getByTestId("graph-locations-card");
    expect(
      within(card).getByRole("group", { name: /^model\.cluster locations$/i }),
    ).toBeInTheDocument();
    expect(
      within(card).getByRole("button", {
        name: /reveal model\.cluster coordinate \(1, 1, 1\)/i,
      }),
    ).toBeInTheDocument();
  });

  it("shows the 3D cluster button when the neuron model type is selected", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();

    expect(await screen.findByTestId("flow")).toBeInTheDocument();
    const panel = await screen.findByTestId("flow-panel-bottom-right");
    expect(
      within(panel).getByRole("button", { name: /open 3d cluster view/i }),
    ).toBeInTheDocument();
    expect(within(panel).queryByTestId("graph-locations-card")).not.toBeInTheDocument();
  });

  it("opens and closes the 3D cluster popup from the neuron type button", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );

    expect(
      await screen.findByRole("dialog", {
        name: /^3d neuron cluster model\.cluster$/i,
      }),
    ).toBeInTheDocument();
    expect(await screen.findByTestId("mock-cluster-3d-scene")).toHaveAttribute(
      "data-visible-x",
      "2",
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

  it("updates 3D cluster slice controls locally", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );
    const scene = await screen.findByTestId("mock-cluster-3d-scene");
    expect(scene).toHaveAttribute("data-visible-x", "2");
    expect(scene).toHaveAttribute("data-visible-y", "2");
    expect(scene).toHaveAttribute("data-visible-z", "1");

    await user.click(screen.getByRole("button", { name: /hide x slice 2/i }));
    expect(await screen.findByTestId("mock-cluster-3d-scene"))
      .toHaveAttribute("data-visible-x", "1");

    await user.click(screen.getByRole("button", { name: /^isolate$/i }));
    expect(await screen.findByTestId("mock-cluster-3d-scene"))
      .toHaveAttribute("data-visible-y", "1");

    await user.click(screen.getByRole("button", { name: /^reset$/i }));
    expect(await screen.findByTestId("mock-cluster-3d-scene"))
      .toHaveAttribute("data-visible-y", "2");
  });

  it("reveals a mapped 3D coordinate in full graph detail mode", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    expect(screen.queryByTestId("node-model.cluster.neuron_1_1_1"))
      .not.toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );
    await user.click(
      await screen.findByRole("button", {
        name: /3d coordinate \(1, 1, 1\) initial/i,
      }),
    );

    expect(screen.getByRole("radio", { name: /full/i })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(await screen.findByTestId("node-model.cluster.neuron_1_1_1"))
      .toBeInTheDocument();
  });

  it("keeps the 3D popup on the cluster for an unmapped coordinate", async () => {
    installFetchMock({
      modelsResponse: neuronModelsResponse,
      inspectResponse: {
        ...locationInspectResponse,
        nodes: locationInspectResponse.nodes.map((node) =>
          node.id === "model.cluster"
            ? {
                ...node,
                details: {
                  cluster: {
                    ...((node.details as { cluster: Record<string, unknown> })
                      .cluster),
                    instantiated: 2,
                    coordinates: [
                      [1, 1, 1],
                      [2, 2, 1],
                    ],
                  },
                },
              }
            : node,
        ),
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );
    await user.click(
      await screen.findByRole("button", {
        name: /3d coordinate \(2, 2, 1\) grown/i,
      }),
    );

    expect(
      screen.getByRole("dialog", {
        name: /^3d neuron cluster model\.cluster$/i,
      }),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("node-model.cluster.neuron_1_1_1"))
      .not.toBeInTheDocument();
  });

  it("closes the 3D popup when selection moves away from the cluster", async () => {
    installFetchMock({
      inspectResponse: locationInspectResponse,
      modelsResponse: neuronModelsResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /open 3d cluster view/i }),
    );
    expect(
      await screen.findByRole("dialog", {
        name: /^3d neuron cluster model\.cluster$/i,
      }),
    ).toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: /^select model$/i }));

    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", {
          name: /^3d neuron cluster model\.cluster$/i,
        }),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps selected cluster locations visible when switching detail modes", async () => {
    installFetchMock({ inspectResponse: locationInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^select model\.cluster$/i }));
    let card = await screen.findByTestId("graph-locations-card");
    await user.click(within(card).getByRole("button", { name: /show cluster locations/i }));
    card = await screen.findByTestId("graph-locations-card");
    const clusterDiagram = await screen.findByTestId("cluster-diagram-model.cluster");
    expect(within(clusterDiagram).getByText("Cluster map")).toBeInTheDocument();
    expect(within(clusterDiagram).getByText("1 / 4")).toBeInTheDocument();
    expect(
      within(clusterDiagram).getByLabelText(/Neuron \(1, 1, 1\).*active/i),
    ).toBeInTheDocument();
    expect(
      await within(card).findByRole("group", { name: /^model\.cluster locations$/i }),
    ).toBeInTheDocument();
    expect(
      within(card).queryByRole("group", {
        name: /^model\.cluster\.terminal locations$/i,
      }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /full/i }));

    await waitFor(() => {
      expect(
        within(card).getByRole("group", { name: /^model\.cluster locations$/i }),
      ).toBeInTheDocument();
    });
    expect(
      within(card).queryByRole("group", {
        name: /^model\.cluster\.terminal locations$/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("reveals a graph node from a location and keeps node details selected", async () => {
    installFetchMock({ inspectResponse: locationInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^select model\.cluster$/i }));
    let card = await screen.findByTestId("graph-locations-card");
    await user.click(within(card).getByRole("button", { name: /show cluster locations/i }));
    card = await screen.findByTestId("graph-locations-card");
    const cluster = await within(card).findByRole("group", {
      name: /^model\.cluster locations$/i,
    });
    await user.click(
      within(cluster).getByRole("button", {
        name: /reveal model\.cluster coordinate \(1, 1, 1\)/i,
      }),
    );

    expect(await screen.findByTestId("node-model.cluster")).toBeInTheDocument();
    expect(await screen.findByTestId("cluster-diagram-model.cluster")).toBeInTheDocument();
    expect(within(card).getByRole("button", { name: /hide cluster locations/i }))
      .toBeInTheDocument();
  });

  it("displays selected node metadata", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
    expect(screen.getByText("33,024")).toBeInTheDocument();
    expect(screen.getByText("BEFORE")).toBeInTheDocument();
  });

});
