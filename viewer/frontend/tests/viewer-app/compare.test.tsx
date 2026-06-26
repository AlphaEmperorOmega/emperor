import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  installFetchMock,
  logRunsResponse,
  renderViewer,
  resetViewerAppTestState,
} from "./support";

function buildCompareLogFixture() {
  const [baseRun] = logRunsResponse.runs;
  const runs = [
    {
      ...baseRun,
      id: "compare-test-linear-layer",
      group: "test_linear_layer",
      experiment: "test_linear_layer",
      modelType: "linears",
      model: "linear",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "test_linear_layer_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      version: "version_0",
      relativePath:
        "test_linear_layer/linear/BASELINE/Mnist/test_linear_layer_20260601_010203/version_0",
      metrics: { "test/accuracy": 0.81 },
    },
    {
      ...baseRun,
      id: "compare-kaggle-linears-all",
      group: "kaggle_linears_all",
      experiment: "kaggle_linears_all",
      modelType: "experts",
      model: "experts_linear",
      preset: "KAGGLE_EXPERT",
      dataset: "KaggleDigits",
      runName: "kaggle_linears_all_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      version: "version_0",
      relativePath:
        "kaggle_linears_all/experts_linear/KAGGLE_EXPERT/KaggleDigits/kaggle_linears_all_20260601_020304/version_0",
      metrics: { "test/accuracy": 0.88 },
    },
  ];

  return {
    logRunsResponse: { runs },
    logExperimentsResponse: {
      experiments: [
        {
          experiment: "kaggle_linears_all",
          runCount: 1,
          relativePath: "kaggle_linears_all",
        },
        {
          experiment: "test_linear_layer",
          runCount: 1,
          relativePath: "test_linear_layer",
        },
      ],
    },
    logTagsByRun: {
      "compare-test-linear-layer": ["validation/accuracy", "train/loss"],
      "compare-kaggle-linears-all": ["validation/accuracy", "train/loss"],
    },
    logScalarSeries: [
      {
        runId: "compare-test-linear-layer",
        tag: "validation/accuracy",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.61 },
          { step: 2, wallTime: 1780000100, value: 0.78 },
        ],
      },
      {
        runId: "compare-kaggle-linears-all",
        tag: "validation/accuracy",
        points: [
          { step: 1, wallTime: 1780000200, value: 0.64 },
          { step: 2, wallTime: 1780000300, value: 0.86 },
        ],
      },
      {
        runId: "compare-test-linear-layer",
        tag: "train/loss",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.71 },
          { step: 2, wallTime: 1780000100, value: 0.33 },
        ],
      },
      {
        runId: "compare-kaggle-linears-all",
        tag: "train/loss",
        points: [
          { step: 1, wallTime: 1780000200, value: 0.68 },
          { step: 2, wallTime: 1780000300, value: 0.28 },
        ],
      },
    ],
  };
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function optionName(label: string) {
  return new RegExp(`^${escapeRegExp(label)}(?:\\b|$)`, "i");
}

async function openCompareScalarTags(user: ReturnType<typeof userEvent.setup>) {
  const trigger = await screen.findByRole("combobox", {
    name: /^Scalar Tags\b/i,
  });
  if (trigger.getAttribute("aria-expanded") !== "true") {
    await user.click(trigger);
  }
  return screen.findByRole("listbox", { name: "Scalar Tags options" });
}

async function selectCompareFixtureRuns(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: /^compare$/i }));
  await screen.findByRole("heading", { name: /training run comparison/i });
  await screen.findByText("Run Target 1");
  await user.click(await screen.findByRole("button", { name: /add target/i }));
  await screen.findByText("Run Target 2");
}

describe("ViewerApp Compare Workspace", () => {
  beforeEach(resetViewerAppTestState);

  it("keeps compare content in a constrained scrollport after main menu navigation", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^compare$/i }));

    const heading = await screen.findByRole("heading", {
      name: /training run comparison/i,
    });
    const scrollRoot = heading.closest(".overflow-y-auto");

    expect(scrollRoot).toBeInstanceOf(HTMLElement);
    expect(scrollRoot).toHaveClass("h-full", "min-h-0", "overflow-y-auto");
    expect(scrollRoot?.parentElement).toHaveClass("h-full", "overflow-hidden");
  });

  it("adds experiment target cards and renders graph/data views", async () => {
    const fixture = buildCompareLogFixture();
    const { logScalarRequests } = installFetchMock(fixture);
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^compare$/i }));
    await screen.findByRole("heading", { name: /training run comparison/i });
    await screen.findByText("Run Target 1");
    await user.click(await screen.findByRole("button", { name: /add target/i }));

    expect(await screen.findByText("Run Target 1")).toBeInTheDocument();
    expect(screen.getByText("Run Target 2")).toBeInTheDocument();
    expect(screen.getAllByText("test_linear_layer").length).toBeGreaterThan(0);
    expect(screen.getAllByText("kaggle_linears_all").length).toBeGreaterThan(0);
    expect(screen.queryByRole("dialog", { name: /add run target/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("Direct Training Run Search")).not.toBeInTheDocument();
    const targetCards = screen.getAllByText(/^Run Target \d$/).map((label) => {
      const card = label.closest("article");
      expect(card).toBeInstanceOf(HTMLElement);
      return card as HTMLElement;
    });
    expect(within(targetCards[0]).getByRole("combobox", { name: "Model Type" }))
      .toHaveTextContent("linears");
    expect(within(targetCards[0]).getByRole("combobox", { name: "Model" }))
      .toHaveTextContent("linear");
    expect(within(targetCards[1]).getByRole("combobox", { name: "Model Type" }))
      .toHaveTextContent("experts");
    expect(within(targetCards[1]).getByRole("combobox", { name: "Model" }))
      .toHaveTextContent("experts_linear");

    expect(
      await screen.findByRole("img", {
        name: /validation\/accuracy training run comparison chart/i,
      }),
    ).toBeInTheDocument();
    expect((await screen.findAllByText("2 lines · 4 points")).length)
      .toBeGreaterThan(0);
    await waitFor(() => {
      expect(logScalarRequests).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            runIds: expect.arrayContaining([
              "compare-test-linear-layer",
              "compare-kaggle-linears-all",
            ]),
            tags: expect.arrayContaining(["validation/accuracy", "train/loss"]),
          }),
        ]),
      );
    });

    await user.click(screen.getByRole("tab", { name: /^data$/i }));

    expect(await screen.findByText("Metric Summary")).toBeInTheDocument();
    expect(screen.getAllByText("Best step").length).toBeGreaterThan(0);
    expect(screen.queryByText("Run Plan")).not.toBeInTheDocument();
  });

  it("keeps many Compare scalar tags compact and searchable", async () => {
    const fixture = buildCompareLogFixture();
    const manyTags = [
      "validation/accuracy",
      "train/loss",
      ...Array.from({ length: 24 }, (_, index) =>
        `debug/scalar_${String(index + 1).padStart(2, "0")}`,
      ),
    ];
    fixture.logTagsByRun = {
      "compare-test-linear-layer": manyTags,
      "compare-kaggle-linears-all": manyTags,
    };
    installFetchMock(fixture);
    renderViewer();
    const user = userEvent.setup();

    await selectCompareFixtureRuns(user);

    expect(await screen.findByText("Run Target 1")).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /add run target/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("Direct Training Run Search")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /^Scalar Tags\b/i }))
      .toBeInTheDocument();
    expect(screen.queryByLabelText(/select metric debug\/scalar_24/i))
      .not.toBeInTheDocument();
    expect(screen.queryByText("debug/scalar_24")).not.toBeInTheDocument();

    const listbox = await openCompareScalarTags(user);
    const dropdownRoot = listbox.closest(".relative");
    expect(dropdownRoot).toBeInstanceOf(HTMLElement);
    await user.type(
      within(dropdownRoot as HTMLElement).getByRole("searchbox", {
        name: /search scalar tags/i,
      }),
      "scalar_24",
    );

    expect(
      await within(listbox).findByRole("option", {
        name: optionName("debug/scalar_24"),
      }),
    ).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("option", {
        name: optionName("debug/scalar_01"),
      }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: /validation\/accuracy training run comparison chart/i,
      }),
    ).toBeInTheDocument();
  });
});
