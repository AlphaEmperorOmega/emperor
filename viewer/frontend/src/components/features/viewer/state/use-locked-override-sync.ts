import { type Dispatch, type SetStateAction, useEffect } from "react";
import { type ConfigField } from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

// When the config schema changes, drop any overrides for fields that have become
// locked (their value is dictated by the schema and is no longer user-editable).
export function useLockedOverrideSync(
  schemaData: { fields: ConfigField[] } | undefined,
  setOverrides: Dispatch<SetStateAction<OverrideValues>>,
) {
  useEffect(() => {
    if (!schemaData?.fields.length) {
      return;
    }
    const lockedKeys = new Set(
      schemaData.fields.filter((field) => field.locked).map((field) => field.key),
    );
    if (lockedKeys.size === 0) {
      return;
    }
    setOverrides((current) => {
      const next = { ...current };
      let changed = false;
      for (const key of lockedKeys) {
        if (Object.prototype.hasOwnProperty.call(next, key)) {
          delete next[key];
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [schemaData, setOverrides]);
}
