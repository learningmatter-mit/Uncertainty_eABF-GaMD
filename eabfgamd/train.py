from copy import deepcopy
from typing import List


__all__ = [
    "make_config_for_each_network",
]


def make_config_for_each_network(orig_params: dict) -> List[dict]:
    """
    Create one config for each network within an ensemble.
    The gist of this function is to remove list from the model keys and use
    only individual values for each network.
    is True.
    """
    parameters = []
    for i in range(orig_params["model"]["num_networks"]):
        params = deepcopy(orig_params)
        params["model"] = {}

        for key, val in orig_params["model"].items():
            if isinstance(val, list) and key not in ["output_keys", "grad_keys"]:
                params["model"][key] = val[i]
            else:
                params["model"][key] = val

        params['dset']['random_state'] = orig_params['dset']['random_state'][i]

        # Set num_networks to 1 for each network
        params["model"]["num_networks"] = 1

        parameters.append(params)

    return parameters
