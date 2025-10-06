interface LatencyBadgeProps {
  latency: number;
}

export const LatencyBadge = ({ latency }: LatencyBadgeProps) => {
  const getColor = (ms: number) => {
    if (ms < 500) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    if (ms < 1500) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${getColor(
        latency
      )}`}
    >
      <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
        <path
          fillRule="evenodd"
          d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
          clipRule="evenodd"
        />
      </svg>
      {latency}ms
    </span>
  );
};
