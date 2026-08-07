"use client";
import { useState } from "react";
import { useJobs, useJobStats } from "@/shared/api/hooks";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";
import { Skeleton } from "@/shared/ui/skeleton";
import { Button } from "@/shared/ui/button";
import { Clock, RefreshCw } from "lucide-react";

const statusVariant: Record<string, "default" | "secondary" | "destructive" | "success" | "warning"> = {
  pending: "warning",
  running: "default",
  done: "success",
  failed: "destructive",
};

function formatDuration(start: string | null, end: string | null) {
  if (!start) return "-";
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  const ms = e - s;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

export function JobsPage() {
  const [page, setPage] = useState(0);
  const pageSize = 50;
  const { data: jobsData, isLoading: jobsLoading, refetch } = useJobs({ limit: pageSize, offset: page * pageSize });
  const { data: statsData } = useJobStats();

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Jobs</h1>
          <p className="text-muted-foreground">Background task monitoring</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" /> Refresh
        </Button>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{statsData?.total ?? "-"}</div>
            <p className="text-xs text-muted-foreground">Total</p>
          </CardContent>
        </Card>
        {["pending", "running", "done", "failed"].map((s) => (
          <Card key={s}>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold">{statsData?.by_status?.[s] ?? 0}</div>
              <p className="text-xs text-muted-foreground capitalize">{s}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Jobs table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" /> Recent Jobs
          </CardTitle>
        </CardHeader>
        <CardContent>
          {jobsLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Request ID</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Error</TableHead>
                    <TableHead>Created</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {jobsData?.jobs?.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                        No jobs found
                      </TableCell>
                    </TableRow>
                  )}
                  {jobsData?.jobs?.map((job) => (
                    <TableRow key={job.id}>
                      <TableCell className="font-mono text-xs">{job.id}</TableCell>
                      <TableCell>
                        <Badge variant="secondary">{job.job_type}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={statusVariant[job.status] ?? "default"}>{job.status}</Badge>
                      </TableCell>
                      <TableCell className="font-mono text-xs max-w-[120px] truncate">
                        {job.request_id ?? "-"}
                      </TableCell>
                      <TableCell className="text-xs">
                        {formatDuration(job.started_at, job.finished_at)}
                      </TableCell>
                      <TableCell className="text-xs text-destructive max-w-[200px] truncate">
                        {job.error_message ?? "-"}
                      </TableCell>
                      <TableCell className="text-xs">
                        {job.creation_date ? new Date(job.creation_date).toLocaleString() : "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <div className="flex justify-between items-center mt-4">
                <span className="text-sm text-muted-foreground">
                  Showing {page * pageSize + 1}-{Math.min((page + 1) * pageSize, jobsData?.total ?? 0)} of{" "}
                  {jobsData?.total ?? 0}
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page === 0}
                    onClick={() => setPage((p) => p - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={(page + 1) * pageSize >= (jobsData?.total ?? 0)}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
