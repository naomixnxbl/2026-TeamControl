"""Generate a bounded Voronoi map PNG for local inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from TeamControl.world.map.voronoi_generator import generate_bounded_voronoi_map
from TeamControl.world.map.voronoi_generator import VoronoiObstacle
from TeamControl.world.map.voronoi_generator import _inset_bounds
from TeamControl.world.map.voronoi_plot import save_voronoi_map_plot
from TeamControl.world.field_config import FIELD_LENGTH_MM, FIELD_WIDTH_MM


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a bounded Voronoi navigation map to a PNG file.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Number of virtual sites for random mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("density_grid", "grid", "random"),
        default="density_grid",
        help="Virtual node placement mode.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=10.0,
        help="Grid density from 10 to 100 percent.",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=120.0,
        help="Minimum guaranteed edge clearance in millimeters.",
    )
    parser.add_argument(
        "--inset",
        type=float,
        default=100.0,
        help="Navigation boundary inset from the real field boundary in millimeters.",
    )
    parser.add_argument(
        "--field-length",
        type=float,
        default=FIELD_LENGTH_MM,
        help="Field length in millimeters. Use the SSL-Vision value for comparison.",
    )
    parser.add_argument(
        "--field-width",
        type=float,
        default=FIELD_WIDTH_MM,
        help="Field width in millimeters. Use the SSL-Vision value for comparison.",
    )
    parser.add_argument(
        "--grid-spacing-x",
        type=float,
        default=None,
        help="Explicit X spacing for grid mode, in millimeters.",
    )
    parser.add_argument(
        "--grid-spacing-y",
        type=float,
        default=None,
        help="Explicit Y spacing for grid mode, in millimeters.",
    )
    parser.add_argument(
        "--max-density-nodes",
        type=int,
        default=240,
        help="Practical cap for density_grid mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed for repeatable maps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "png" / "bounded_voronoi_map.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--obstacle",
        action="append",
        default=None,
        help="Obstacle as x,y,radius[,label]. Can be passed multiple times.",
    )
    parser.add_argument(
        "--random-obstacles",
        type=int,
        default=0,
        help="Generate this many random circular obstacles inside the navigation bounds.",
    )
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=240.0,
        help="Fixed random-obstacle radius in millimeters.",
    )
    parser.add_argument(
        "--obstacle-radius-min",
        type=float,
        default=None,
        help="Minimum random-obstacle radius. Overrides --obstacle-radius with max.",
    )
    parser.add_argument(
        "--obstacle-radius-max",
        type=float,
        default=None,
        help="Maximum random-obstacle radius. Overrides --obstacle-radius with min.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a Matplotlib window after saving.",
    )
    return parser.parse_args(argv)


def parse_obstacles(values) -> tuple[VoronoiObstacle, ...]:
    obstacles = []
    for index, value in enumerate(values or ()):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in (3, 4):
            raise ValueError("Obstacle must be x,y,radius[,label]")
        label = parts[3] if len(parts) == 4 else f"obs{index}"
        obstacles.append(
            VoronoiObstacle(
                pos_mm=(float(parts[0]), float(parts[1])),
                radius_mm=float(parts[2]),
                label=label,
            )
        )
    return tuple(obstacles)


def random_obstacles(args: argparse.Namespace) -> tuple[VoronoiObstacle, ...]:
    if args.random_obstacles <= 0:
        return ()
    if args.obstacle_radius <= 0:
        raise ValueError("--obstacle-radius must be positive")

    radius_min = args.obstacle_radius_min
    radius_max = args.obstacle_radius_max
    if radius_min is None and radius_max is None:
        radius_min = radius_max = args.obstacle_radius
    elif radius_min is None:
        radius_min = args.obstacle_radius
    elif radius_max is None:
        radius_max = args.obstacle_radius
    if radius_min <= 0 or radius_max <= 0 or radius_min > radius_max:
        raise ValueError("random obstacle radius range must be positive and ordered")

    field_bounds = (
        -args.field_length / 2.0,
        args.field_length / 2.0,
        -args.field_width / 2.0,
        args.field_width / 2.0,
    )
    nav_bounds = _inset_bounds(field_bounds, args.inset)
    rng = random.Random(args.seed)
    obstacles = []
    for index in range(args.random_obstacles):
        radius = rng.uniform(radius_min, radius_max)
        x_min, x_max, y_min, y_max = nav_bounds
        if radius * 2.0 >= min(x_max - x_min, y_max - y_min):
            raise ValueError("random obstacle radius is too large for navigation bounds")
        x = rng.uniform(x_min + radius, x_max - radius)
        y = rng.uniform(y_min + radius, y_max - radius)
        obstacles.append(
            VoronoiObstacle(
                pos_mm=(x, y),
                radius_mm=radius,
                label=f"rand{index}",
            )
        )
    return tuple(obstacles)


def main() -> None:
    args = parse_args()
    obstacles = parse_obstacles(args.obstacle) + random_obstacles(args)
    voronoi_map = generate_bounded_voronoi_map(
        args.nodes,
        field_length_mm=args.field_length,
        field_width_mm=args.field_width,
        min_clearance_mm=args.clearance,
        boundary_inset_mm=args.inset,
        placement_mode=args.mode,
        density_percent=args.density,
        grid_spacing_x_mm=args.grid_spacing_x,
        grid_spacing_y_mm=args.grid_spacing_y,
        max_density_nodes=args.max_density_nodes,
        obstacles=obstacles,
        seed=args.seed,
    )
    output_path = save_voronoi_map_plot(
        voronoi_map,
        args.output,
        show=args.show,
    )

    print(f"Saved Voronoi map PNG: {output_path}")
    print(
        f"Generated {len(voronoi_map.virtual_sites_mm)} virtual sites, "
        f"{len(voronoi_map.obstacles)} obstacles, "
        f"{len(voronoi_map.cells)} cells, "
        f"{len(voronoi_map.edges)} safe edges."
    )
    print(
        f"Mode={voronoi_map.placement_mode}, "
        f"field={voronoi_map.field_bounds_mm[1] - voronoi_map.field_bounds_mm[0]:.0f}"
        f"x{voronoi_map.field_bounds_mm[3] - voronoi_map.field_bounds_mm[2]:.0f} mm, "
        f"inset={voronoi_map.boundary_inset_mm:.0f} mm, "
        f"clearance={voronoi_map.min_clearance_mm:.0f} mm."
    )


if __name__ == "__main__":
    main()
