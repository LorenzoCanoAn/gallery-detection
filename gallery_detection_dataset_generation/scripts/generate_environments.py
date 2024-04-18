from argparse import ArgumentParser
import os
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    TunnelNetworkParams,
    GrownTunnelGenerationParams,
    ConnectorTunnelGenerationParams,
)
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
    TunnelPtClGenParams,
)
import numpy as np


class Args:
    def __init__(self):
        args = self.get_args()
        self.n_environments = args.n_environments
        self.save_folder = args.save_folder
        self.n_grown_tunnels = args.n_grown_tunnels
        self.n_connector_tunnels = args.n_connector_tunnels
        self.min_r = args.min_r
        self.max_r = args.max_r
        self.min_l = args.min_l
        self.max_l = args.max_l
        self.htend_deg = args.htend_deg
        self.vtend_deg = args.vtend_deg
        self.min_sl = args.min_sl
        self.max_sl = args.max_sl
        self.htend_deg = args.htend_deg
        self.vtend_deg = args.vtend_deg
        self.hnoise_deg = args.hnoise_deg
        self.vnoise_deg = args.vnoise_deg
        self.min_fta = args.min_fta
        self.max_fta = args.max_fta

    def get_args(self):
        parser = ArgumentParser("generate_environments")
        parser.add_argument("--n_environments", type=int, default=10)
        parser.add_argument(
            "--save_folder", type=str, default="/home/lorenzo/gazebo_worlds/training_worlds/"
        )
        parser.add_argument("--n_grown_tunnels", type=int, default=4)
        parser.add_argument("--n_connector_tunnels", type=int, default=4)
        parser.add_argument("--min_r", type=float, required=False, default=2)
        parser.add_argument("--max_r", type=float, required=False, default=7)
        parser.add_argument("--min_l", type=float, required=False, default=50)
        parser.add_argument("--max_l", type=float, required=False, default=100)
        parser.add_argument("--min_sl", type=float, required=False, default=10)
        parser.add_argument("--max_sl", type=float, required=False, default=20)
        parser.add_argument("--min_fta", type=float, required=False, default=-2)
        parser.add_argument("--max_fta", type=float, required=False, default=-1)
        parser.add_argument("--htend_deg", type=float, required=False, default=20)
        parser.add_argument("--vtend_deg", type=float, required=False, default=10)
        parser.add_argument("--hnoise_deg", type=float, required=False, default=5)
        parser.add_argument("--vnoise_deg", type=float, required=False, default=5)
        return parser.parse_args()


def setup_save_folder(path_to_save_folder, n_envs_to_gen):
    if os.path.isdir(path_to_save_folder):
        response = input("The save directory already exists. [(O)verwrite/(A)ppend] (default A)?")
        if response.lower() in ["append", "", "a"]:
            wif = os.listdir(path_to_save_folder)
            n_wif = len(wif)
            start_n = n_wif
        elif response.lower() in ["overwrite", "o"]:
            os.system(f"rm -rf {path_to_save_folder}/**")
            start_n = 0
        else:
            raise Exception(f"Invalid option: {response}")
    else:
        os.makedirs(path_to_save_folder)
        start_n = 0
    end_n = start_n + n_envs_to_gen
    dirpaths = []
    for n_world in range(start_n, end_n):
        dirname = f"env_{n_world:03d}"
        dirpath = os.path.join(path_to_save_folder, dirname)
        os.makedirs(dirpath, exist_ok=True)
        dirpaths.append(dirpath)
    return dirpaths


def generate_network(args: Args):
    tn = TunnelNetwork(params=TunnelNetworkParams.from_defaults())
    for n in range(args.n_grown_tunnels):
        params = GrownTunnelGenerationParams.random()
        params.from_defaults()
        params.distance = np.random.uniform(args.min_l, args.max_l)
        params.horizontal_tendency = np.deg2rad(np.random.uniform(-args.htend_deg, args.htend_deg))
        params.horizontal_noise = np.deg2rad(np.random.uniform(0, args.hnoise_deg))
        params.vertical_tendency = np.deg2rad(np.random.uniform(-args.vtend_deg, args.vtend_deg))
        params.vertical_noise = np.deg2rad(np.random.uniform(0, args.vnoise_deg))
        params.min_segment_length = args.min_sl
        params.max_segment_length = args.max_sl
        tn.add_random_grown_tunnel(params, n_trials=100)
    for n in range(args.n_connector_tunnels):
        params = ConnectorTunnelGenerationParams.random()
        params.node_position_horizontal_noise = 0
        params.node_position_vertical_noise = 0
        tn.add_random_connector_tunnel(params, n_trials=100)
    return tn


def generate_mesh(args: Args, tn: TunnelNetwork):
    TunnelPtClGenParams._random_radius_interval = (args.min_r, args.max_r)
    TunnelPtClGenParams._random_noise_multiplier_interval = (0, 0.3)
    ptcl_params = TunnelNetworkPtClGenParams.random()
    mesh_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_params.fta_distance = np.random.uniform(args.min_fta, args.max_fta)
    tnmg = TunnelNetworkMeshGenerator(tn, ptcl_gen_params=ptcl_params, meshing_params=mesh_params)
    tnmg.compute_all()
    return tnmg


def gen_axis_points_file(mesh_generator: TunnelNetworkMeshGenerator):
    axis_points = np.zeros((0, 3 + 3 + 1 + 1 + 1))
    for tunnel in mesh_generator._tunnel_network.tunnels:
        radius = mesh_generator.ptcl_params_of_tunnel(tunnel).radius
        aps = mesh_generator.aps_of_tunnels
        avs = mesh_generator.avs_of_tunnels
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        tunnel_flags = np.ones((len(aps), 1)) * 1
        tunnel_id = np.ones((len(aps), 1)) * hash(tunnel)
        axis_points = np.concatenate(
            (axis_points, np.concatenate((aps, avs, rds, tunnel_flags, tunnel_id), axis=1)), axis=0
        )
    for intersection in mesh_generator._tunnel_network.intersections:
        for tunnel in mesh_generator._tunnel_network._tunnels_of_node[intersection]:
            radius = mesh_generator.ptcl_params_of_tunnel(tunnel).radius
            aps = mesh_generator._aps_avs_of_intersections[intersection][tunnel][:, 0:3]
            avs = mesh_generator._aps_avs_of_intersections[intersection][tunnel][:, 3:6]
            assert len(aps) == len(avs)
            if len(aps) == 0:
                continue
            rds = np.ones((len(aps), 1)) * radius
            intersection_flag = np.ones((len(aps), 1)) * 2
            tunnel_id = np.ones((len(aps), 1)) * hash(tunnel)
            axis_points = np.concatenate(
                (
                    axis_points,
                    np.concatenate((aps, avs, rds, intersection_flag, tunnel_id), axis=1),
                ),
                axis=0,
            )
    return axis_points


def main():
    args = Args()
    dirpaths = setup_save_folder(args.save_folder, args.n_environments)
    for dirpath in dirpaths:
        meshpath = os.path.join(dirpath, "mesh.obj")
        axispath = os.path.join(dirpath, "axis.txt")
        ftadpath = os.path.join(dirpath, "fta_dist.txt")
        modelpath = os.path.join(dirpath, "model.sdf")
        tn = generate_network(args)
        tnmg = generate_mesh(args, tn)
        tnmg.save_mesh(meshpath)
        np.savetxt(axispath, gen_axis_points_file(tnmg))
        np.savetxt(ftadpath, np.array((tnmg._meshing_params.fta_distance,)))


if __name__ == "__main__":
    main()
