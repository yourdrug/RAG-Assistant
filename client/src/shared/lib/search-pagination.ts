export function getSearchPagination(total: number, offset: number, limit: number) {
  const pages = Math.ceil(total / limit);
  return {
    page: Math.floor(offset / limit) + 1,
    pages,
    hasPrevious: offset > 0,
    hasNext: offset + limit < total,
    lastOffset: Math.max(0, (pages - 1) * limit),
  };
}
