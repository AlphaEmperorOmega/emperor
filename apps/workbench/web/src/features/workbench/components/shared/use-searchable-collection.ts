import {
  type RefCallback,
  type UIEventHandler,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

const DEFAULT_ROW_HEIGHT = 44;
const DEFAULT_VIEWPORT_HEIGHT = 264;
const DEFAULT_OVERSCAN_ROWS = 3;

export function searchableCollectionKey<Option>(
  options: Option[],
  optionKey: (option: Option) => string,
) {
  return options.map(optionKey).join("\u0000");
}

type VirtualViewport = Readonly<{
  scrollTop: number;
  height: number;
}>;

export type SearchableVirtualItem<Option> = Readonly<{
  index: number;
  key: string;
  option: Option;
  start: number;
  size: number;
}>;

export function useSearchableCollectionWindow<
  Option,
  ElementType extends HTMLElement = HTMLDivElement,
>({
  options,
  optionKey,
  estimatedRowHeight = DEFAULT_ROW_HEIGHT,
  fallbackViewportHeight = DEFAULT_VIEWPORT_HEIGHT,
  overscanRows = DEFAULT_OVERSCAN_ROWS,
  onScrollIndexChange,
}: {
  options: Option[];
  optionKey: (option: Option) => string;
  estimatedRowHeight?: number;
  fallbackViewportHeight?: number;
  overscanRows?: number;
  onScrollIndexChange?: (index: number) => void;
}) {
  const collectionElementRef = useRef<ElementType | null>(null);
  const collectionResizeObserverRef = useRef<ResizeObserver | null>(null);
  const rowResizeObserverRef = useRef<ResizeObserver | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const pendingViewportRef = useRef<VirtualViewport | null>(null);
  const [viewport, setViewport] = useState<VirtualViewport>({
    scrollTop: 0,
    height: fallbackViewportHeight,
  });
  const [measuredHeights, setMeasuredHeights] = useState<
    ReadonlyMap<string, number>
  >(new Map());

  const layout = useMemo(() => {
    const offsets = new Array<number>(options.length + 1);
    const sizes = new Array<number>(options.length);
    offsets[0] = 0;
    for (let index = 0; index < options.length; index += 1) {
      const key = optionKey(options[index]);
      const size = measuredHeights.get(key) ?? estimatedRowHeight;
      sizes[index] = size;
      offsets[index + 1] = offsets[index] + size;
    }
    return { offsets, sizes, totalSize: offsets[options.length] ?? 0 };
  }, [estimatedRowHeight, measuredHeights, optionKey, options]);

  const updateViewport = useCallback((next: VirtualViewport) => {
    setViewport((current) =>
      current.scrollTop === next.scrollTop && current.height === next.height
        ? current
        : next,
    );
  }, []);

  const indexAtOffset = useCallback(
    (offset: number) => {
      let index = 0;
      while (
        index < options.length - 1 &&
        layout.offsets[index + 1] <= offset
      ) {
        index += 1;
      }
      return options.length > 0 ? index : -1;
    },
    [layout.offsets, options.length],
  );

  const collectionRef = useCallback<RefCallback<ElementType>>(
    (element) => {
      collectionResizeObserverRef.current?.disconnect();
      collectionResizeObserverRef.current = null;
      collectionElementRef.current = element;
      if (!element) {
        return;
      }

      updateViewport({
        scrollTop: element.scrollTop,
        height: element.clientHeight || fallbackViewportHeight,
      });
      if (typeof ResizeObserver === "function") {
        const observer = new ResizeObserver(() => {
          updateViewport({
            scrollTop: element.scrollTop,
            height: element.clientHeight || fallbackViewportHeight,
          });
        });
        observer.observe(element);
        collectionResizeObserverRef.current = observer;
      }
    },
    [fallbackViewportHeight, updateViewport],
  );

  const measureOption = useCallback<RefCallback<HTMLElement>>((element) => {
    if (!element) {
      return;
    }
    const key = element.dataset.virtualOptionKey;
    if (!key) {
      return;
    }

    const recordHeight = () => {
      const height =
        element.getBoundingClientRect().height ||
        element.offsetHeight ||
        estimatedRowHeight;
      setMeasuredHeights((current) => {
        if (current.get(key) === height) {
          return current;
        }
        const next = new Map(current);
        next.set(key, height);
        return next;
      });
    };
    recordHeight();
    if (typeof ResizeObserver === "function") {
      rowResizeObserverRef.current ??= new ResizeObserver((entries) => {
        setMeasuredHeights((current) => {
          let next: Map<string, number> | null = null;
          for (const entry of entries) {
            const row = entry.target as HTMLElement;
            const rowKey = row.dataset.virtualOptionKey;
            if (!rowKey) {
              continue;
            }
            const height =
              entry.borderBoxSize?.[0]?.blockSize ||
              entry.contentRect.height ||
              row.offsetHeight ||
              estimatedRowHeight;
            if ((next ?? current).get(rowKey) !== height) {
              next ??= new Map(current);
              next.set(rowKey, height);
            }
          }
          return next ?? current;
        });
      });
      rowResizeObserverRef.current.observe(element);
    }
  }, [estimatedRowHeight]);

  useEffect(
    () => () => {
      collectionResizeObserverRef.current?.disconnect();
      rowResizeObserverRef.current?.disconnect();
      if (scrollFrameRef.current !== null) {
        window.cancelAnimationFrame(scrollFrameRef.current);
      }
      collectionResizeObserverRef.current = null;
      rowResizeObserverRef.current = null;
      scrollFrameRef.current = null;
      pendingViewportRef.current = null;
      collectionElementRef.current = null;
    },
    [],
  );

  const handleCollectionScroll = useCallback<UIEventHandler<ElementType>>(
    (event) => {
      const element = event.currentTarget;
      pendingViewportRef.current = {
        scrollTop: element.scrollTop,
        height: element.clientHeight || fallbackViewportHeight,
      };
      if (scrollFrameRef.current !== null) {
        return;
      }
      scrollFrameRef.current = window.requestAnimationFrame(() => {
        scrollFrameRef.current = null;
        const pending = pendingViewportRef.current;
        pendingViewportRef.current = null;
        if (pending) {
          updateViewport(pending);
          onScrollIndexChange?.(indexAtOffset(pending.scrollTop));
        }
      });
    },
    [
      fallbackViewportHeight,
      indexAtOffset,
      onScrollIndexChange,
      updateViewport,
    ],
  );

  const scrollToIndex = useCallback(
    (index: number, align: "nearest" | "start" | "end" = "nearest") => {
      if (index < 0 || index >= options.length) {
        return;
      }
      const start = layout.offsets[index];
      const end = layout.offsets[index + 1];
      const element = collectionElementRef.current;
      const height = element?.clientHeight || viewport.height;
      const currentTop = element?.scrollTop ?? viewport.scrollTop;
      let nextTop = currentTop;
      if (align === "start") {
        nextTop = start;
      } else if (align === "end") {
        nextTop = Math.max(0, end - height);
      } else if (start < currentTop) {
        nextTop = start;
      } else if (end > currentTop + height) {
        nextTop = Math.max(0, end - height);
      }
      if (nextTop === currentTop) {
        return;
      }
      if (element) {
        element.scrollTop = nextTop;
      }
      updateViewport({ scrollTop: nextTop, height });
    },
    [layout.offsets, options.length, updateViewport, viewport.height, viewport.scrollTop],
  );

  const resetCollectionScroll = useCallback(() => {
    const element = collectionElementRef.current;
    if (element) {
      element.scrollTop = 0;
    }
    updateViewport({
      scrollTop: 0,
      height: element?.clientHeight || fallbackViewportHeight,
    });
  }, [fallbackViewportHeight, updateViewport]);

  let virtualWindow: {
    startIndex: number;
    endIndex: number;
    beforeHeight: number;
    afterHeight: number;
    virtualOptions: SearchableVirtualItem<Option>[];
  };
  if (options.length === 0) {
    virtualWindow = {
      startIndex: 0,
      endIndex: 0,
      beforeHeight: 0,
      afterHeight: 0,
      virtualOptions: [],
    };
  } else {
    const overscan = estimatedRowHeight * overscanRows;
    const visibleStart = Math.max(0, viewport.scrollTop - overscan);
    const visibleEnd = viewport.scrollTop + viewport.height + overscan;
    let startIndex = 0;
    while (
      startIndex < options.length &&
      layout.offsets[startIndex + 1] < visibleStart
    ) {
      startIndex += 1;
    }
    let endIndex = startIndex;
    while (
      endIndex < options.length &&
      layout.offsets[endIndex] <= visibleEnd
    ) {
      endIndex += 1;
    }

    const virtualOptions = options
      .slice(startIndex, endIndex)
      .map((option, offset) => {
        const index = startIndex + offset;
        return {
          index,
          key: optionKey(option),
          option,
          start: layout.offsets[index],
          size: layout.sizes[index],
        };
      });
    virtualWindow = {
      startIndex,
      endIndex,
      beforeHeight: layout.offsets[startIndex],
      afterHeight: Math.max(
        0,
        layout.totalSize - layout.offsets[endIndex],
      ),
      virtualOptions,
    };
  }

  return {
    collectionRef,
    measureOption,
    handleCollectionScroll,
    scrollToIndex,
    resetCollectionScroll,
    totalHeight: layout.totalSize,
    ...virtualWindow,
  };
}
