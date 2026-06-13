import * as echarts from "echarts/core";
import { BarChart, CustomChart, LineChart, TreemapChart } from "echarts/charts";
import {
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

// Register only the chart types, components, and renderer the viewer actually
// uses. Importing through `echarts/core` keeps `echarts-for-react` from pulling
// the full bundle.
echarts.use([
  LineChart,
  BarChart,
  CustomChart,
  TreemapChart,
  GridComponent,
  TooltipComponent,
  AxisPointerComponent,
  DataZoomComponent,
  LegendComponent,
  MarkLineComponent,
  CanvasRenderer,
]);

export { echarts };
