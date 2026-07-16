export type ImplementedFeature = {
  id: string;
  category: string;
  title: string;
  description: string;
};

export const IMPLEMENTED_FEATURES: ImplementedFeature[] = [
  {
    id: "api-health-cors",
    category: "Backend API and discovery",
    title: "API health and CORS",
    description: "Reports API availability and allows configured local workbench origins.",
  },
  {
    id: "model-discovery",
    category: "Backend API and discovery",
    title: "Model discovery",
    description: "Lists inspectable model packages for the target selector.",
  },
  {
    id: "model-type-filtering",
    category: "Backend API and discovery",
    title: "Model type filtering",
    description: "Groups public model IDs by family and filters target model choices.",
  },
  {
    id: "preset-discovery",
    category: "Backend API and discovery",
    title: "Preset discovery",
    description: "Loads model presets with labels and descriptions after model selection.",
  },
  {
    id: "dataset-discovery",
    category: "Backend API and discovery",
    title: "Dataset discovery",
    description: "Finds model datasets and exposes labels plus input and output dimensions.",
  },
  {
    id: "monitor-discovery",
    category: "Backend API and discovery",
    title: "Monitor discovery",
    description: "Returns available monitor options, kinds, descriptions, and defaults.",
  },
  {
    id: "config-schema-extraction",
    category: "Backend API and discovery",
    title: "Config schema extraction",
    description: "Extracts config fields, choices, defaults, nullable state, and preset locks.",
  },
  {
    id: "override-parsing-locked-rejection",
    category: "Backend API and discovery",
    title: "Override parsing and locked override rejection",
    description: "Parses submitted override values and rejects invalid or preset-locked fields.",
  },
  {
    id: "model-inspection",
    category: "Inspection and serialization",
    title: "Model inspection",
    description: "Builds the selected model and returns an inspectable module graph.",
  },
  {
    id: "backend-graph-serialization",
    category: "Inspection and serialization",
    title: "Backend graph serialization",
    description: "Serializes module nodes, containment edges, roles, parameters, and metadata.",
  },
  {
    id: "cli-inspection",
    category: "Inspection and serialization",
    title: "CLI inspection",
    description: "Provides text and JSON inspection from the backend command line.",
  },
  {
    id: "frontend-api-client",
    category: "Frontend state and config",
    title: "Frontend API client",
    description: "Wraps workbench API calls with typed helpers, validation, and error extraction.",
  },
  {
    id: "frontend-target-query-state",
    category: "Frontend state and config",
    title: "Frontend target and query state",
    description: "Coordinates selected targets, datasets, monitors, overrides, and preview queries.",
  },
  {
    id: "config-summary-full-dialog",
    category: "Frontend state and config",
    title: "Config summary and full config dialog",
    description: "Shows config sections, editable fields, override state, and training commands.",
  },
  {
    id: "graph-canvas-modes-scopes-layout",
    category: "Graph visualization",
    title: "Graph canvas, modes, scopes, and layout",
    description: "Renders the graph with detail modes, opened or entire scopes, and layout.",
  },
  {
    id: "graph-card-analysis-views",
    category: "Graph visualization",
    title: "Graph card analysis views",
    description: "Displays summaries, mechanisms, diagrams, parameter shapes, and details.",
  },
  {
    id: "structure-rail",
    category: "Graph visualization",
    title: "Structure rail",
    description: "Shows the model tree and reveals hidden graph nodes on demand.",
  },
  {
    id: "location-terminal-reach-panels",
    category: "Graph visualization",
    title: "Location and terminal reach panels",
    description: "Summarizes neuron cluster coordinates and terminal reach for selected nodes.",
  },
  {
    id: "selected-node-details-monitor-entry",
    category: "Graph visualization",
    title: "Selected node details and monitor entry",
    description: "Shows selected node metadata and opens monitor charts after training exists.",
  },
  {
    id: "training-job-creation-polling-cancellation",
    category: "Training and monitor data",
    title: "Training job creation, polling, and cancellation",
    description: "Starts jobs, polls status and logs, displays metrics, and cancels runs.",
  },
  {
    id: "training-worker-progress-events",
    category: "Training and monitor data",
    title: "Training worker and progress events",
    description: "Runs training in a worker process and streams progress events to the UI.",
  },
  {
    id: "tensorboard-monitor-data",
    category: "Training and monitor data",
    title: "TensorBoard monitor data",
    description: "Reads matching TensorBoard scalars, histograms, and images for a node.",
  },
  {
    id: "ui-primitives",
    category: "UI foundation",
    title: "UI primitives",
    description: "Provides shared buttons, badges, inputs, switches, selects, and cards.",
  },
];
