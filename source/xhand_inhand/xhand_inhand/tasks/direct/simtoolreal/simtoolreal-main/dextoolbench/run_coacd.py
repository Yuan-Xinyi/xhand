from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import trimesh
import tyro


def run_coacd(
    mesh_path: Path,
    output_dir: Path,
    max_convex_hull: int = -1,
    mode: Literal["subprocess", "python"] = "python",
) -> List[trimesh.Trimesh]:
    """Run COACD on a mesh and return the list of convex hulls."""
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"
    assert mesh_path.suffix == ".obj", f"Mesh file {mesh_path} is not an OBJ file"

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "subprocess":
        from subprocess import run

        output_mesh_path = output_dir / mesh_path.name
        cmd = f"coacd -i {mesh_path} -o {output_mesh_path} -c {max_convex_hull}"
        print(f"Running command: {cmd}")
        run(cmd, shell=True, check=True)

        assert output_mesh_path.exists(), (
            f"Output mesh file {output_mesh_path} does not exist"
        )
        output_mesh = trimesh.load(output_mesh_path)
        parts = output_mesh.split()
        for i, part in enumerate(parts):
            filename = output_dir / f"decomp_{i}.obj"
            print(f"Saving part {i} to {filename}")
            part.export(filename)
        print(
            f"Decomposition complete. {len(parts)} parts found and saved to {output_dir}."
        )
        return parts
    elif mode == "python":
        import coacd
        import numpy as np

        input_mesh = trimesh.load(mesh_path, force="mesh")
        coacd_mesh = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
        convex_vs_fs_parts = coacd.run_coacd(
            coacd_mesh, max_convex_hull=max_convex_hull
        )
        parts = []
        for vs, fs in convex_vs_fs_parts:
            parts.append(trimesh.Trimesh(vs, fs))

        np.random.seed(0)
        scene = trimesh.Scene()
        for part in parts:
            part.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(
                np.uint8
            )
            scene.add_geometry(part)
        output_mesh_path = output_dir / mesh_path.name
        print(f"Saving scene to {output_mesh_path}")
        scene.export(output_mesh_path)

        for i, part in enumerate(parts):
            filename = output_dir / f"decomp_{i}.obj"
            print(f"Saving part {i} to {filename}")
            part.export(filename)
        print(
            f"Decomposition complete. {len(parts)} parts found and saved to {output_dir}."
        )
        return parts
    else:
        raise ValueError(f"Invalid mode: {mode}")


@dataclass
class RunCoacdArgs:
    mesh_path: Path
    """Path to the input OBJ mesh file."""

    output_dir: Path
    """Directory to save the decomposed convex hull parts."""

    max_convex_hull: int = -1
    """Maximum number of convex hulls (-1 for unlimited)."""

    mode: Literal["subprocess", "python"] = "python"
    """COACD execution mode: 'subprocess' calls the coacd CLI, 'python' uses the Python API."""


def main():
    args: RunCoacdArgs = tyro.cli(RunCoacdArgs)
    run_coacd(
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        max_convex_hull=args.max_convex_hull,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
