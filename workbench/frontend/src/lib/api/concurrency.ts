export async function mapWithConcurrency<TItem, TResult>(
  items: readonly TItem[],
  concurrency: number,
  worker: (item: TItem, index: number) => Promise<TResult>,
): Promise<TResult[]> {
  if (items.length === 0) {
    return [];
  }
  const limit = Math.max(1, Math.floor(concurrency));
  const results = new Array<TResult>(items.length);
  let nextIndex = 0;
  let activeCount = 0;
  let settled = false;

  return new Promise<TResult[]>((resolve, reject) => {
    const launch = () => {
      if (settled) {
        return;
      }
      if (nextIndex >= items.length && activeCount === 0) {
        settled = true;
        resolve(results);
        return;
      }
      while (activeCount < limit && nextIndex < items.length) {
        const index = nextIndex;
        nextIndex += 1;
        activeCount += 1;
        worker(items[index], index).then(
          (result) => {
            activeCount -= 1;
            results[index] = result;
            launch();
          },
          (error: unknown) => {
            settled = true;
            reject(error);
          },
        );
      }
    };

    launch();
  });
}
