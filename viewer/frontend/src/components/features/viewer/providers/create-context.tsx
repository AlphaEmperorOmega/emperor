import { createContext, useContext, type ReactNode } from "react";

/**
 * Builds a typed React context whose consumer hook throws a clear error when
 * used outside its provider, removing the repetitive null-check boilerplate.
 */
export function createViewerContext<TValue>(displayName: string) {
  const Context = createContext<TValue | null>(null);
  Context.displayName = displayName;

  function Provider({ value, children }: { value: TValue; children: ReactNode }) {
    return <Context.Provider value={value}>{children}</Context.Provider>;
  }
  Provider.displayName = `${displayName}.Provider`;

  function useContextValue(): TValue {
    const value = useContext(Context);
    if (value === null) {
      throw new Error(`${displayName} value is missing; render within its provider.`);
    }
    return value;
  }

  return [Provider, useContextValue] as const;
}
