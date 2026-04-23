"""
to be written:
Event bus for connecting modules without direct imports
"""

class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_name, callback):
        self._subscribers.setdefault(event_name, []).append(callback)

    def publish(self, event_name, data=None):
        for cb in self._subscribers.get(event_name, []):
            cb(data)

bus = EventBus()

# # ──────────────────────────────────────────────────────────────────
# # 2) Subscribe handlers (do NOT start sim here!)
# # ──────────────────────────────────────────────────────────────────

# def on_trial_start(data):
#     print(f"  [bus] Trial {data['trial_index']} started | pose={data['pose']} | seed={data['seed']}")

# def on_trial_result(data):
#     print(f"  [bus] Trial {data['trial_index']} result | success={data['success']} | attempts={data['attempt_number']}")

# def on_trial_ended(data):
#     print(f"  [bus] Trial {data['trial_index']} ended")

# def on_session_ended(data):
#     print(f"\n[bus] Session finished | success={data['success_count']}/{data['total_attempts']}")


# bus.subscribe("trial_start", on_trial_start)
# bus.subscribe("trial_result", on_trial_result)
# bus.subscribe("trial_ended", on_trial_ended)
# bus.subscribe("session_ended", on_session_ended)
# print("  [bus] 4 event handlers attached (will run after you start sim)")


# # ═══════════════════════════════════════════════════════════════
# # 3. NEW: SCRIPT STATE (Track what's happening)
# # ═══════════════════════════════════════════════════════════════

# class ScriptState:
#     """
#     Tracks global script execution state:
#       - is_running    : whether sim is active
#       - current_trial : {trial_index, seed, stage} or None
#       - session_stats : {total_attempts, success_count, ...}
#     """
#     def __init__(self):
#         self.is_running       = False
#         self.current_trial    = None   # dict with:
#                                       # { "trial_index": int,
#                                       #   "seed": int,
#                                       #   "stage": "train|val|test" }
#         self.session_stats    = {
#             "total_attempts":     0,
#             "success_count":      0,
#             "failures_no_object": 0,
#             "failures_physics":   0,
#             "failures_other":     0,
#         }

#     def reset(self):
#         """Reset stats at start of new session."""
#         self.session_stats = {
#             "total_attempts":     0,
#             "success_count":      0,
#             "failures_no_object": 0,
#             "failures_physics":   0,
#             "failures_other":     0,
#         }


# state = ScriptState()

# print("  [state] ScriptState class defined")


# # ═══════════════════════════════════════════════════════════════
# # Event publisher helpers (call these from trial_runner)
# # ═══════════════════════════════════════════════════════════════

# def publish_trial_start(trial_index, pose, seed, stage):
#     bus.publish("trial_start", {
#         "trial_index": trial_index,
#         "pose":        pose,
#         "seed":        seed,
#         "stage":       stage,
#     })

# def publish_trial_result(trial_index, success, attempt_number, failures=None):
#     if failures is None:
#         failures = {}

#     bus.publish("trial_result", {
#         "trial_index":       trial_index,
#         "success":           success,
#         "attempt_number":    attempt_number,
#         "failures":          failures,
#     })

# def publish_trial_ended(trial_index, success):
#     bus.publish("trial_ended", {
#         "trial_index": trial_index,
#         "success":   success,
#     })

# def publish_session_ended(success_count, total_attempts):
#     bus.publish("session_ended", {
#         "success_count":  success_count,
#         "total_attempts": total_attempts,
#     })


# print("  [bus] Publisher helpers defined (trial_start, trial_result, trial_ended, session_ended)")


# # ═══════════════════════════════════════════════════════════════
# # 4) Save trial results to JSON per trial
# # ──────────────────────────────────────────────────────────────────

# def _build_trial_output_path(base_dir, trial_index, seed, stage):
#     """Build a structured output path: base_dir/stage/seed/trial_N_seed_M.json"""
#     os.makedirs(os.path.join(base_dir, stage), exist_ok=True)
#     filename = f"trial_{trial_index}_seed_{seed}.json"
#     return os.path.join(base_dir, stage, filename)


# def save_trial_result(base_dir, trial_index, seed, stage, result_dict):
#     """
#     Save trial result to JSON file.
#     result_dict should include all data you want to keep per trial.
#     """
#     path = _build_trial_output_path(base_dir, trial_index, seed, stage)
#     with open(path, 'w') as f:
#         json.dump(result_dict, f, indent=4)
#     return path


# print("  [bus] Trial result saver defined (save_trial_result)")


# # ═══════════════════════════════════════════════════════════════
# # 5) Save session summary (Called after all trials in a stage)
# # ═══════════════════════════════════════════════════════════════

# def save_session_summary(base_dir, stage, session_stats, trial_count, duration_sec):
#     """
#     Save a summary JSON for the stage.
#     session_stats: dict from ScriptState.session_stats
#     trial_count: how many trials were run in this stage
#     duration_sec: total time for this stage
#     """
#     path = os.path.join(base_dir, f"summary_{stage}.json")
#     summary = {
#         "stage":               stage,
#         "trial_count":         trial_count,
#         "total_attempts":      session_stats["total_attempts"],
#         "success_count":       session_stats["success_count"],
#         "failures_no_object":  session_stats["failures_no_object"],
#         "failures_physics":    session_stats["failures_physics"],
#         "failures_other":      session_stats["failures_other"],
#         "success_rate":        (session_stats["success_count"] / trial_count * 100) if trial_count > 0 else 0,
#         "duration_seconds":    duration_sec,
#     }

#     with open(path, 'w') as f:
#         json.dump(summary, f, indent=4)
#     return path


# print("  [bus] Session summary saver defined (save_session_summary)")


# # ═══════════════════════════════════════════════════════════════
# # 6) Save final aggregated report (after train + val + test)
# # ═══════════════════════════════════════════════════════════════

# def save_final_report(base_dir, all_stage_stats, grand_totals, duration_sec):
#     """
#     all_stage_stats: list of session summary dicts (one per stage)
#     grand_totals: dict like
#       {
#         "total_attempts": int,
#         "success_count": int,
#         "failures_no_object": int,
#         "failures_physics": int,
#         "failures_other": int,
#       }
#     duration_sec: total time across all stages
#     """
#     report = {
#         "stages":                  all_stage_stats,
#         "grand_totals":            grand_totals,
#         "overall_success_rate":    (grand_totals["success_count"] / grand_totals["total_attempts"] * 100) if grand_totals["total_attempts"] > 0 else 0,
#         "total_duration_seconds":  duration_sec,
#         "timestamp":               datetime.now().isoformat(),
#     }

#     path = os.path.join(base_dir, "final_report.json")
#     with open(path, 'w') as f:
#         json.dump(report, f, indent=4)

#     print(f"\n[bus] ✅ Final report saved: {path}")
#     return path


# print("  [bus] Final report saver defined (save_final_report)")  

