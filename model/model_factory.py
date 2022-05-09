import sys

model_dict = {}
transfer_dict = {}


def get_model_by_name(net_name, **kwargs):
    return model_dict.get(net_name)(**kwargs)


def transfer_weights(net_name, state_dict, early_stride=4):
    if transfer_dict[net_name] is None:
        raise NotImplementedError
    else:
        return transfer_dict[net_name](state_dict, early_stride)


def remove_fc(net_name, state_dict):
    if net_name.startswith('c2d_eftnet'):
        state_dict.pop('_fc.weight', None)
        state_dict.pop('_fc.bias', None)
        return state_dict
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    if net_name.startswith('lgd3d_') or net_name.startswith('lgd_p3d_'):
        state_dict.pop('fc_g.weight', None)
        state_dict.pop('fc_g.bias', None)
    if net_name.startswith('dg_p3da_') or net_name.startswith('dg_p3d_') or net_name.startswith('dgtc_c2d_'):
        state_dict.pop('fc_dual.weight', None)
        state_dict.pop('fc_dual.bias', None)
    return state_dict


def remove_defcor_weight(net_name, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if ('def_cor.weight' in k):
            continue
        else: new_state_dict[k] = v
    return new_state_dict


# model registration
# from https://github.com/rwightman/pytorch-image-models/
def register_model(fn):
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__

    # add entries to registry dict/sets
    assert model_name not in model_dict
    model_dict[model_name] = fn
    if hasattr(mod, 'transfer_weights'):
        transfer_dict[model_name] = mod.transfer_weights
    else:
        transfer_dict[model_name] = None
    return fn
