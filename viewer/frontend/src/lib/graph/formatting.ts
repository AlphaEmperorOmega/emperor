// Public boundary for node formatting. Implementation is split across the
// `format/` folder by concern (text, badges, height); this module re-exports
// their public surface so existing `@/lib/graph/formatting` imports keep working.
export * from "@/lib/graph/format/text";
export * from "@/lib/graph/format/badges";
export * from "@/lib/graph/format/height";
