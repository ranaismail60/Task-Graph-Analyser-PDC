

from collections import defaultdict
import matplotlib.pyplot as plt


class TaskGraphAnalyser:
    def __init__(self, weights, predecessors, graph_name="Graph"):
        self.name = graph_name
        self.W = weights
        self.PRED = predecessors
        self.tasks = sorted(list(self.W.keys()))

        # Compute successor relationships
        self.SUCC = defaultdict(list)
        for task in self.tasks:
            for pred in self.PRED.get(task, []):
                self.SUCC[pred].append(task)

    def get_start_nodes(self):
        return [t for t in self.tasks if not self.PRED.get(t, [])]

    def get_finish_nodes(self):
        return [t for t in self.tasks if not self.SUCC.get(t, [])]

    def find_all_paths(self):
        start_nodes = self.get_start_nodes()
        finish_nodes = set(self.get_finish_nodes())
        all_paths = []

        def dfs(curr, current_path):
            if curr in finish_nodes:
                length = sum(self.W[node] for node in current_path)
                all_paths.append((list(current_path), length))
                return
            for succ in sorted(self.SUCC.get(curr, [])):
                dfs(succ, current_path + [succ])

        for start in start_nodes:
            dfs(start, [start])

        return all_paths

    def get_critical_path(self):
        paths = self.find_all_paths()
        # Returns path with maximum total weight
        return max(paths, key=lambda x: x[1])

    def get_total_work(self):
        return sum(self.W.values())

    def get_asap_schedule(self):
        asap_start = {}
        asap_finish = {}

        # Topological sort / dependence resolution
        remaining_preds = {t: len(self.PRED.get(t, [])) for t in self.tasks}
        ready = [t for t in self.tasks if remaining_preds[t] == 0]

        while ready:
            ready.sort()
            curr = ready.pop(0)

            if not self.PRED.get(curr, []):
                start_t = 0
            else:
                start_t = max(asap_finish[p] for p in self.PRED[curr])

            asap_start[curr] = start_t
            asap_finish[curr] = start_t + self.W[curr]

            for succ in self.SUCC.get(curr, []):
                remaining_preds[succ] -= 1
                if remaining_preds[succ] == 0:
                    ready.append(succ)

        return asap_start, asap_finish

    def get_max_concurrency(self):
        asap_start, asap_finish = self.get_asap_schedule()
        if not asap_finish:
            return 0, []

        max_time = max(asap_finish.values())
        events = sorted(
            list(
                set(
                    list(asap_start.values())
                    + list(asap_finish.values())
                    + [0, max_time]
                )
            )
        )

        intervals = []
        max_conc = 0

        for i in range(len(events) - 1):
            t_start, t_end = events[i], events[i + 1]
            # Count tasks active in time interval (t_start, t_end)
            active_count = sum(
                1
                for t in self.tasks
                if asap_start[t] <= t_start and asap_finish[t] >= t_end
            )
            if active_count > max_conc:
                max_conc = active_count
                intervals = [(t_start, t_end)]
            elif active_count == max_conc:
                intervals.append((t_start, t_end))

        return max_conc, intervals

    def schedule_p_processors(self, num_processors):
        """
        List scheduling rule:
        Whenever a processor becomes free, pick the ready task with smallest number/name.
        P1 is checked before P2, P2 before P3, etc.
        """
        task_status = {
            t: "UNREADY" for t in self.tasks
        }  # UNREADY, READY, RUNNING, COMPLETED
        for t in self.tasks:
            if not self.PRED.get(t, []):
                task_status[t] = "READY"

        procs = [
            {"task": None, "end_time": 0} for _ in range(num_processors)
        ]  # 0-indexed
        schedule = []  # tuple: (proc_id, task, start_time, end_time)
        completed_tasks = set()
        current_time = 0

        while len(completed_tasks) < len(self.tasks):
            # 1. Complete tasks whose time has arrived
            for p_idx in range(num_processors):
                if (
                    procs[p_idx]["task"] is not None
                    and procs[p_idx]["end_time"] <= current_time
                ):
                    finished_t = procs[p_idx]["task"]
                    completed_tasks.add(finished_t)
                    task_status[finished_t] = "COMPLETED"
                    procs[p_idx]["task"] = None

            # 2. Update READY status for remaining unready tasks
            for t in self.tasks:
                if task_status[t] == "UNREADY":
                    if all(p in completed_tasks for p in self.PRED.get(t, [])):
                        task_status[t] = "READY"

            # 3. Assign ready tasks to free processors checking P1 before P2 ...
            ready_queue = sorted(
                [t for t in self.tasks if task_status[t] == "READY"]
            )

            for p_idx in range(num_processors):
                if procs[p_idx]["task"] is None and ready_queue:
                    next_task = ready_queue.pop(0)
                    start_t = current_time
                    end_t = start_t + self.W[next_task]
                    procs[p_idx] = {"task": next_task, "end_time": end_t}
                    task_status[next_task] = "RUNNING"
                    schedule.append((p_idx + 1, next_task, start_t, end_t))

            # 4. Advance time to next event
            future_times = [
                p["end_time"]
                for p in procs
                if p["task"] is not None and p["end_time"] > current_time
            ]
            if future_times:
                current_time = min(future_times)
            else:
                break

        total_time = max(item[3] for item in schedule) if schedule else 0
        return schedule, total_time

    def plot_gantt_chart(self, num_processors, schedule, critical_path, filename):
        """Bonus: Draw Gantt chart with critical-path tasks highlighted."""
        fig, ax = plt.subplots(figsize=(10, 4))
        cp_set = set(critical_path)

        for proc_id, task, start, end in schedule:
            is_cp = task in cp_set
            color = "#d9534f" if is_cp else "#428bca"  # Red for CP, Blue for non-CP
            ax.barh(
                proc_id,
                end - start,
                left=start,
                align="center",
                color=color,
                edgecolor="black",
                height=0.5,
            )
            ax.text(
                (start + end) / 2,
                proc_id,
                f"{task}\n({self.W[task]})",
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=9,
            )

        ax.set_yticks(range(1, num_processors + 1))
        ax.set_yticklabels([f"P{i}" for i in range(1, num_processors + 1)])
        ax.set_xlabel("Time Units")
        ax.set_ylabel("Processors")
        ax.set_title(
            f"Gantt Chart ({self.name}) - p={num_processors} Processors"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def analyze(self):
        print("=" * 70)
        print(f" ANALYSIS REPORT: {self.name}")
        print("=" * 70)

        # 1. Start and Finish Nodes
        print(f"(1) Start Nodes : {self.get_start_nodes()}")
        print(f"    Finish Nodes: {self.get_finish_nodes()}\n")

        # 2. All Paths
        paths = self.find_all_paths()
        print("(2) All Start-to-Finish Paths:")
        for path, length in paths:
            print(f"    {' -> '.join(path)} | Length: {length}")
        print()

        # 3. Critical Path
        cp_path, cp_len = self.get_critical_path()
        print(f"(3) Critical Path       : {' -> '.join(cp_path)}")
        print(f"    Critical Path Length: {cp_len}\n")

        # 4. Total Work and Average Concurrency
        work = self.get_total_work()
        avg_conc = work / cp_len
        print(f"(4) Total Work (W)                 : {work}")
        print(f"    Average Degree of Concurrency  : {avg_conc:.2f}\n")

        # 5. ASAP Schedule and Max Concurrency
        asap_s, asap_f = self.get_asap_schedule()
        print("(5) ASAP Start / Finish Times:")
        for t in self.tasks:
            print(f"    {t:4s}: Start = {asap_s[t]:2d}, Finish = {asap_f[t]:2d}")

        max_conc, intervals = self.get_max_concurrency()
        interval_str = ", ".join([f"[{st}, {en})" for st, en in intervals])
        print(
            f"    Maximum Degree of Concurrency  : {max_conc} (occurring in interval(s): {interval_str})\n"
        )

        # 6. Schedules for p = 1, 2, 3
        print("(6) Processor Scheduling Results:")
        for p in [1, 2, 3]:
            sched, t_p = self.schedule_p_processors(p)
            speedup = work / t_p
            print(f"    --- p = {p} Processor(s) ---")
            print(
                f"    {'Processor':<10} | {'Task':<6} | {'Start':<6} | {'End':<6}"
            )
            print("    " + "-" * 38)
            for proc_id, task, st, en in sched:
                print(f"    P{proc_id:<9} | {task:<6} | {st:<6} | {en:<6}")
            print(f"    Total Time (Tp) : {t_p}")
            print(f"    Speedup (S_p)   : {speedup:.3f}\n")

            # Save Gantt Chart Plot
            filename = f"gantt_{self.name.lower().replace(' ', '_')}_p{p}.png"
            self.plot_gantt_chart(p, sched, cp_path, filename)


# ==============================================================================
# GRAPH DEFINITIONS
# ==============================================================================

# Graph of Question 2
q2_weights = {
    "T1": 8,
    "T2": 6,
    "T3": 3,
    "T4": 5,
    "T5": 3,
    "T6": 7,
    "T7": 3,
    "T8": 5,
    "T9": 2,
}
q2_pred = {
    "T1": [],
    "T2": [],
    "T3": [],
    "T4": ["T1"],
    "T5": ["T1", "T2"],
    "T6": ["T2", "T3"],
    "T7": ["T4", "T5"],
    "T8": ["T6"],
    "T9": ["T7", "T8"],
}

# Graph 3a: ((Islamabad OR Multan) AND 2022) AND IT
q3a_weights = {"N1": 12, "N2": 12, "N3": 12, "N4": 12, "NA": 10, "NB": 18, "NC": 13}
q3a_pred = {
    "N1": [],
    "N2": [],
    "N3": [],
    "N4": [],  # N1: Isb, N2: Mlt, N3: 2022, N4: IT
    "NA": ["N1", "N2"],
    "NB": ["NA", "N3"],
    "NC": ["NB", "N4"],
}

# Graph 3b: (IT AND 2022) AND (Islamabad OR Multan)
q3b_weights = {"N1": 12, "N2": 12, "N3": 12, "N4": 12, "NX": 14, "NY": 10, "NZ": 14}
q3b_pred = {
    "N1": [],
    "N2": [],
    "N3": [],
    "N4": [],  # N1: IT, N2: 2022, N3: Isb, N4: Mlt
    "NX": ["N1", "N2"],
    "NY": ["N3", "N4"],
    "NZ": ["NX", "NY"],
}

if __name__ == "__main__":
    analyser_q2 = TaskGraphAnalyser(q2_weights, q2_pred, "Question 2 Graph")
    analyser_q2.analyze()

    analyser_q3a = TaskGraphAnalyser(
        q3a_weights, q3a_pred, "Question 3a Graph"
    )
    analyser_q3a.analyze()

    analyser_q3b = TaskGraphAnalyser(
        q3b_weights, q3b_pred, "Question 3b Graph"
    )
    analyser_q3b.analyze()