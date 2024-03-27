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
)
import numpy as np


def get_args():
    parser = ArgumentParser("generate_environments")
    parser.add_argument("--n_tunnels", type=int, required=False, default=5)
    parser.add_argument("--n_environments", type=int, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--n_grown_tunnels", type=int, required=True)
    parser.add_argument("--n_connector_tunnels", type=int, required=True)
    parser.add_argument("--max_r", type=float, required=False, default=7)
    parser.add_argument("--min_r", type=float, required=False, default=2)
    parser.add_argument("--max_l", type=float, required=False, default=50)
    parser.add_argument("--min_l", type=float, required=False, default=20)
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


def generate_network(args):
    tn = TunnelNetwork(params=TunnelNetworkParams.from_defaults())
    for n in range(args.n_grown_tunnels):
        params = GrownTunnelGenerationParams.from_defaults()
        params.from_defaults()
        params.distance = np.random.uniform(args.min_l, args.max_l)
        params.horizontal_tendency = np.deg2rad(np.random.uniform(-20, 20))
        tn.add_random_grown_tunnel(params, n_trials=100)
    for n in range(args.n_connector_tunnels):
        tn.add_random_connector_tunnel(n_trials=100)
    return tn


def generate_mesh(args, tn: TunnelNetwork):
    ptcl_params = TunnelNetworkPtClGenParams.from_defaults()
    tnmg = TunnelNetworkMeshGenerator(tn, ptcl_gen_params=ptcl_params)


def main():
    args = get_args()
    dirpaths = setup_save_folder(args.save_folder, args.n_environments)
    for dirpath in dirpaths:
        meshpath = os.path.join(dirpath, "mesh.obj")
        axispath = os.path.join(dirpath, "axis.txt")
        ftadpath = os.path.join(dirpath, "fta_dist.txt")
        modelpath = os.path.join(dirpath, "model.sdf")
        tn = generate_network(args)


if __name__ == "__main__":
    main()
