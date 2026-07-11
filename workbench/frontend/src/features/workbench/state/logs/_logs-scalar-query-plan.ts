import {
  DEFAULT_LOG_SCALAR_MAX_POINTS,
  LOG_SCALAR_SAMPLING,
} from "@/lib/api";
import { type LogScalarQueryInput } from "@/features/workbench/state/logs/use-log-queries";
import { logQueryKeys } from "@/lib/query-keys";

export const LOG_SCALAR_TAG_CHUNK_SIZE = 6;
// Ten runs keeps the first progressive response bounded while cutting the
// common 100-run/six-tag view from 50 serialized requests to 10.
export const LOG_SCALAR_RUN_CHUNK_SIZE = 10;

export function buildLogScalarQueryInput({
  enabled,
  group,
  selectedTagList,
  visibleRunIds,
}: {
  enabled: boolean;
  group?: string;
  selectedTagList: string[];
  visibleRunIds: string[];
}): LogScalarQueryInput {
  return {
    runIds: visibleRunIds,
    tags: selectedTagList,
    enabled: enabled && visibleRunIds.length > 0 && selectedTagList.length > 0,
    group,
    queryKey: logQueryKeys.scalarsForRunsAndTags(visibleRunIds, selectedTagList, {
      group,
      maxPoints: DEFAULT_LOG_SCALAR_MAX_POINTS,
      sampling: LOG_SCALAR_SAMPLING,
    }),
  };
}

function chunkUniqueValues(values: string[], chunkSize: number) {
  const size = Math.max(1, Math.floor(chunkSize));
  const uniqueValues = Array.from(new Set(values));
  const chunks: string[][] = [];
  for (let index = 0; index < uniqueValues.length; index += size) {
    chunks.push(uniqueValues.slice(index, index + size));
  }
  return chunks;
}

export function chunkScalarTagsForQueries(
  tags: string[],
  chunkSize = LOG_SCALAR_TAG_CHUNK_SIZE,
) {
  return chunkUniqueValues(tags, chunkSize);
}

export function chunkScalarRunIdsForQueries(
  runIds: string[],
  chunkSize = LOG_SCALAR_RUN_CHUNK_SIZE,
) {
  return chunkUniqueValues(runIds, chunkSize);
}

export function buildLogScalarChunkQueryInputs({
  enabled,
  group,
  requestedTags,
  selectedTagList,
  visibleRunIds,
}: {
  enabled: boolean;
  group: string;
  requestedTags: Set<string>;
  selectedTagList: string[];
  visibleRunIds: string[];
}) {
  const requestedSelectedTags = selectedTagList.filter((tag) =>
    requestedTags.has(tag),
  );
  const tagChunks = chunkScalarTagsForQueries(requestedSelectedTags);
  const runChunks = chunkScalarRunIdsForQueries(visibleRunIds);
  return runChunks.flatMap((runIds) =>
    tagChunks.map((tags) =>
      buildLogScalarQueryInput({
        enabled,
        group,
        selectedTagList: tags,
        visibleRunIds: runIds,
      }),
    ),
  );
}
