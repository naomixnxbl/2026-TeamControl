#!/usr/bin/env python

import argparse
import sys
import time
from multiprocessing import Process, Queue, Event, freeze_support

from TeamControl.process_workers.vision_runner import VisionProcess
from TeamControl.process_workers.gcfsm_runner import GCfsm
from TeamControl.process_workers.wm_runner import WMWorker
from TeamControl.process_workers.robot_recv_runner import RobotRecv
from TeamControl.world.model_manager import WorldModelManager

from TeamControl.utils.Logger import LogSaver
from TeamControl.dispatcher.dispatch import Dispatcher
from TeamControl.utils.yaml_config import Config
from TeamControl.utils.sim_config import Btv2Config, Sim3v3Config, Sim6v6Config
from TeamControl.onboard_vision import build_ip_map

from TeamControl.robot.goalie import run_goalie
from TeamControl.robot.striker import run_striker
from TeamControl.robot.navigator import run_navigator, WAYPOINTS_A, WAYPOINTS_B
from TeamControl.robot.voronoi_game_navigator import run_voronoi_game_navigator
from TeamControl.robot.team import run_team
from TeamControl.robot.coop import run_coop
from TeamControl.bt.run_bt_v2_process import run_bt_v2_process


def _wait_for_vision(wm, timeout=15.0):
    """Block until the world model has received at least one vision frame."""
    deadline = time.time() + timeout
    print("[main] Waiting for first vision frame...", flush=True)
    while time.time() < deadline:
        frame = wm.get_latest_frame()
        if frame is not None:
            n_yellow = sum(1 for _ in frame.robots_yellow)
            n_blue   = sum(1 for _ in frame.robots_blue)
            ball     = frame.ball
            ball_str = f"({ball.x:.0f}, {ball.y:.0f}) mm" if ball is not None else "not visible"
            print(
                f"[main] Vision confirmed — "
                f"yellow robots: {n_yellow}, blue robots: {n_blue}, ball: {ball_str}",
                flush=True,
            )
            return True
        time.sleep(0.05)
    print(f"[main] ERROR: No vision data after {timeout:.0f}s — is grSim running and broadcasting?", flush=True)
    return False


def main():
    freeze_support()
    parser = argparse.ArgumentParser(
        description="RoboCup SSL Team Control — multi-mode launcher",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "calibration",
            "vision_only",
            "voronoi_test",
            "goalie",
            "1v1",
            "obstacle",
            "coop",
            "6v6",
            "btv2",
            "btv2_test",
            "bt_3v3",
            "bt_6v6",
        ],
        default="calibration",
        help=(
            "calibration  — backend only; no robot behaviour\n"
            "vision_only  — run vision + world model only, no robots\n"
            "voronoi_test — one robot per team uses Voronoi planning\n"
            "goalie       — yellow goalie vs blue striker\n"
            "1v1          — yellow striker vs blue striker\n"
            "obstacle     — two robots chasing ball with obstacle avoidance\n"
            "coop         — two robots cooperate to score (pass + shoot)\n"
            "6v6          — full 6v6 match (1 goalie + 5 field per team)\n"
            "btv2         — BT v2 pipeline (config from sim_config.py)\n"
            "btv2_test    — single robot via BT v2 coordinator\n"
            "bt_3v3       — 3v3 both teams via BT v2 (config from sim_config.py)\n"
            "bt_6v6       — 6v6 both teams via BT v2 (config from sim_config.py)"
        ),
    )
    parser.add_argument(
        "--skip-gc",
        action="store_true",
        help="Skip the Game Controller process (useful for testing without a live GC)",
    )
    args = parser.parse_args()

    preset = Config()
    our_id    = preset.goalie_yellow_id or 0
    enemy_id  = preset.goalie_blue_id   or 0

    # ── Queues ────────────────────────────────────────────────────────────────
    vision_q   = Queue()
    gc_q       = Queue()
    dispatch_q = Queue()
    recv_q     = Queue()
    ip_map     = build_ip_map(preset)

    logger = None

    # ── Shared state ──────────────────────────────────────────────────────────
    is_running = Event()

    wm_manager = WorldModelManager()
    wm_manager.start()
    wm = wm_manager.WorldModel(us_yellow=preset.us_yellow, us_positive=preset.us_positive)

    # ── Background processes (always needed) ──────────────────────────────────
    background = [
        Process(target=VisionProcess.run_worker,
                args=(is_running, logger, vision_q,
                      preset.use_grSim_vision, preset.vision[1]),
                name="VisionProcess"),
        Process(target=WMWorker.run_worker,
                args=(is_running, logger, wm, vision_q, gc_q, recv_q, ip_map),
                name="WMWorker"),
        Process(target=Dispatcher.run_worker,
                args=(is_running, logger, dispatch_q, preset),
                name="Dispatcher"),
        Process(target=RobotRecv.run_worker,
                args=(is_running, logger, preset.robot_ip, recv_q)),
    ]
    if not args.skip_gc:
        background.append(Process(
            target=GCfsm.run_worker,
            args=(is_running, logger, gc_q,
                  preset.us_yellow, preset.us_positive, preset.team_name),
            name="GCfsm",
        ))
    else:
        print("[main] --skip-gc: Game Controller process skipped", flush=True)

    # ── Mode-specific foreground processes ────────────────────────────────────
    foreground = []

    if args.mode in ("calibration", "vision_only"):
        pass  # no robot behaviour processes

    elif args.mode == "voronoi_test":
        foreground.append(Process(target=run_voronoi_game_navigator,
                                  args=(is_running, dispatch_q, wm, our_id, preset.us_yellow)))
        foreground.append(Process(target=run_voronoi_game_navigator,
                                  args=(is_running, dispatch_q, wm, enemy_id, not preset.us_yellow)))

    elif args.mode == "goalie":
        foreground.append(Process(target=run_goalie,
                                  args=(is_running, dispatch_q, wm, our_id, preset.us_yellow)))
        foreground.append(Process(target=run_striker,
                                  args=(is_running, dispatch_q, wm, enemy_id, not preset.us_yellow)))

    elif args.mode == "1v1":
        foreground.append(Process(target=run_striker,
                                  args=(is_running, dispatch_q, wm, our_id, True)))
        foreground.append(Process(target=run_striker,
                                  args=(is_running, dispatch_q, wm, enemy_id, False)))

    elif args.mode == "obstacle":
        foreground.append(Process(target=run_navigator,
                                  args=(is_running, dispatch_q, wm, our_id,
                                        preset.us_yellow, WAYPOINTS_A)))
        foreground.append(Process(target=run_navigator,
                                  args=(is_running, dispatch_q, wm, enemy_id,
                                        preset.us_yellow, WAYPOINTS_B)))

    elif args.mode == "coop":
        us_y    = preset.us_yellow
        enemy_y = not us_y
        foreground.append(Process(target=run_coop,
                                  args=(is_running, dispatch_q, wm, our_id, enemy_id, us_y),
                                  kwargs=dict(mate_is_yellow=enemy_y, attack_positive=True)))
        foreground.append(Process(target=run_coop,
                                  args=(is_running, dispatch_q, wm, enemy_id, our_id, enemy_y),
                                  kwargs=dict(mate_is_yellow=us_y, attack_positive=True)))

    elif args.mode == "6v6":
        foreground.append(Process(target=run_team,
                                  args=(is_running, dispatch_q, wm, True, our_id)))
        foreground.append(Process(target=run_team,
                                  args=(is_running, dispatch_q, wm, False, enemy_id)))

    elif args.mode == "btv2":
        sim = Btv2Config()
        foreground.append(Process(
            target=run_bt_v2_process,
            args=(is_running, wm, dispatch_q),
            kwargs=dict(
                is_yellow=sim.controlled_is_yellow,
                robot_ids=sim.controlled_robot_ids,
                role_assignment=sim.roles,
                heuristic_role_swap=sim.heuristic_role_swap,
                movement_safety=sim.movement_safety,
                tick_period=sim.tick_period,
            ),
            daemon=True,
            name=f"btv2_{sim.controlled_team}",
        ))

    elif args.mode == "btv2_test":
        foreground.append(Process(
            target=run_bt_v2_process,
            args=(is_running, wm, dispatch_q),
            kwargs=dict(
                is_yellow=preset.us_yellow,
                robot_ids=[our_id],
            ),
            daemon=True,
        ))

    elif args.mode in ("bt_3v3", "bt_6v6"):
        sim = Sim3v3Config() if args.mode == "bt_3v3" else Sim6v6Config()
        for is_yellow, robot_ids in ((True, sim.yellow_ids), (False, sim.blue_ids)):
            foreground.append(Process(
                target=run_bt_v2_process,
                args=(is_running, wm, dispatch_q),
                kwargs=dict(
                    is_yellow=is_yellow,
                    robot_ids=robot_ids,
                    role_assignment=sim.roles,
                    heuristic_role_swap=sim.heuristic_role_swap,
                    movement_safety=sim.movement_safety,
                    tick_period=sim.tick_period,
                ),
                daemon=True,
                name=f"{args.mode}_{'yellow' if is_yellow else 'blue'}",
            ))

    # ── Start background, wait for vision, then start robots ──────────────────
    is_running.set()
    print(f"[main] Starting mode: {args.mode}", flush=True)

    for p in background:
        p.start()

    if not _wait_for_vision(wm, timeout=15.0):
        print("[main] Aborting — no vision data.", flush=True)
        is_running.clear()
        for p in background:
            p.join(timeout=5)
        sys.exit(1)

    for p in foreground:
        p.start()

    # ── Main loop: watchdog + exit prompt ─────────────────────────────────────
    while is_running.is_set():
        for p in background:
            if not p.is_alive():
                print(f"[main] CRITICAL: process '{p.name}' died — shutting down", flush=True)
                is_running.clear()
                break

        try:
            print("Type 'exit' to quit: ")
            user_input = input()
            if user_input.lower() == "exit":
                print("Shutdown signal received...")
                is_running.clear()
                break
        except KeyboardInterrupt:
            print("\nShutdown signal received...")
            is_running.clear()

        time.sleep(1)

    # ── Join all processes ────────────────────────────────────────────────────
    for p in foreground:
        p.join(timeout=5)
    for p in background:
        p.join(timeout=5)

    print("All processes have been ended")


if __name__ == "__main__":
    main()
