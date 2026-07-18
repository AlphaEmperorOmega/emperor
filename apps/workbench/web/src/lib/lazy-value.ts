type LazyFunction = (...args: never[]) => unknown;

export function createLazyValue<Value>(load: () => Promise<Value>) {
  let cachedValue: Value | undefined;
  let hasCachedValue = false;
  let pendingLoad: Promise<Value> | null = null;

  return function loadValue(): Promise<Value> {
    if (hasCachedValue) {
      return Promise.resolve(cachedValue as Value);
    }
    if (pendingLoad) {
      return pendingLoad;
    }

    let loaded: Promise<Value>;
    try {
      loaded = load();
    } catch (error) {
      return Promise.reject(error);
    }
    const loadPromise = loaded.then(
      (value) => {
        cachedValue = value;
        hasCachedValue = true;
        pendingLoad = null;
        return value;
      },
      (error: unknown) => {
        pendingLoad = null;
        throw error;
      },
    );
    pendingLoad = loadPromise;
    return loadPromise;
  };
}

export function createLazyFunction<Implementation extends LazyFunction>(
  load: () => Promise<Implementation>,
) {
  const loadImplementation = createLazyValue(load);
  return async (
    ...args: Parameters<Implementation>
  ): Promise<Awaited<ReturnType<Implementation>>> => {
    const implementation = await loadImplementation();
    return implementation(...args) as Awaited<ReturnType<Implementation>>;
  };
}
