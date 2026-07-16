import { describe, expect, it } from "vitest";
import {
  eventMetadataSchema,
  responseCompletenessSchema,
} from "@/lib/api/schemas";

describe("shared API response metadata schemas", () => {
  it("accepts complete, truncated, and omitted response completeness metadata", () => {
    expect(responseCompletenessSchema.parse({})).toEqual({});
    expect(
      responseCompletenessSchema.parse({
        sourceItemCount: 14,
        returnedItemCount: 10,
        truncated: true,
        truncationReason: "limit",
      }),
    ).toEqual({
      sourceItemCount: 14,
      returnedItemCount: 10,
      truncated: true,
      truncationReason: "limit",
    });
  });

  it("extends completeness metadata with event-file observations", () => {
    expect(
      eventMetadataSchema.parse({
        eventBytes: 2048,
        skippedEventFiles: 2,
        sourceItemCount: null,
        returnedItemCount: 0,
        truncated: false,
        truncationReason: null,
      }),
    ).toEqual({
      eventBytes: 2048,
      skippedEventFiles: 2,
      sourceItemCount: null,
      returnedItemCount: 0,
      truncated: false,
      truncationReason: null,
    });
  });
});
