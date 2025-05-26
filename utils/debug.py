import torch


def get_layers(m):
    # This function will get layers recursively. Only the basic layer(e.g. Linear, Softmax) will be added to the
    # list, which means ModuleList, Sequential and layers that contain other layers will not be added.
    layers = list(m.children())
    flatt_layers = []
    if not layers:
        return m
    else:
        for layer in layers:
            try:
                flatt_layers.extend(get_layers(layer))
            except TypeError:
                flatt_layers.append(get_layers(layer))
    return flatt_layers


def check_layer_input_range_fp_hook(module, layer_input, layer_output):
    maximum = torch.max(layer_input[0])
    minimum = torch.min(layer_input[0])
    module_name = module.__class__.__name__
    if torch.isnan(maximum) or torch.isnan(minimum) or torch.isinf(maximum) or torch.isnan(minimum):
        module.check_layer_input_range_msg = f'[Error] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
    else:
        module.check_layer_input_range_msg = f'[Info] {module_name}  min: {minimum.item()}  max: {maximum.item()}'


def check_layer_output_range_fp_hook(module, layer_input, layer_output):
    maximum = torch.max(layer_output[0])
    minimum = torch.min(layer_output[0])
    module_name = module.__class__.__name__
    if torch.isnan(maximum) or torch.isnan(minimum) or torch.isinf(maximum) or torch.isnan(minimum):
        module.check_layer_output_range_msg = f'[Error] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
    else:
        module.check_layer_output_range_msg = f'[Info] {module_name}  min: {minimum.item()}  max: {maximum.item()}'


def check_layer_parameter_range_fp_hook(module, layer_input, layer_output):
    params = list(module.parameters())
    module_name = module.__class__.__name__
    if params:
        max_list = []
        min_list = []
        for each_param in params:
            max_list.append(torch.max(each_param))
            min_list.append(torch.min(each_param))
        maximum = max(max_list)
        minimum = min(min_list)
        if torch.isnan(maximum) or torch.isnan(minimum) or torch.isinf(maximum) or torch.isnan(minimum):
            module.check_layer_parameter_range_msg = f'[Error] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
        else:
            module.check_layer_parameter_range_msg = f'[Info] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
    else:
        module.check_layer_parameter_range_msg = f'[Warning] {module_name}  there are not parameters in this layer'


def check_gradient_input_range_bp_hook(module, grad_input, grad_output):
    module_name = module.__class__.__name__
    if isinstance(grad_input[0], torch.Tensor):
        maximum = torch.max(grad_input[0])
        minimum = torch.min(grad_input[0])
        if torch.isnan(maximum) or torch.isnan(minimum) or torch.isinf(maximum) or torch.isnan(minimum):
            module.check_gradient_input_range_msg = f'[Error] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
        else:
            module.check_gradient_input_range_msg = f'[Info] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
    else:
        module.check_gradient_input_range_msg = f'[Warning] {module_name}  this layer is connected to input data'


def check_gradient_output_range_bp_hook(module, grad_input, grad_output):
    maximum = torch.max(grad_output[0])
    minimum = torch.min(grad_output[0])
    module_name = module.__class__.__name__
    if torch.isnan(maximum) or torch.isnan(minimum) or torch.isinf(maximum) or torch.isnan(minimum):
        module.check_gradient_output_range_msg = f'[Error] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
    else:
        module.check_gradient_output_range_msg = f'[Info] {module_name}  min: {minimum.item()}  max: {maximum.item()}'


def check_gradient_parameter_range_bp_hook(module, grad_input, grad_output):
    module_name = module.__class__.__name__
    params = list(module.parameters())
    if params:
        max_list = []
        min_list = []
        for each_param in params:
            if not isinstance(each_param.grad, torch.Tensor):
                max_list.append(torch.tensor(100000))
                min_list.append(torch.tensor(-100000))
            else:
                max_list.append(torch.max(each_param.grad))
                min_list.append(torch.min(each_param.grad))
        maximum = max(max_list)
        minimum = min(min_list)
        if torch.isnan(maximum) or torch.isnan(minimum) or torch.isinf(maximum) or torch.isnan(minimum):
            module.check_gradient_parameter_range_msg = f'[Error] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
        else:
            module.check_gradient_parameter_range_msg = f'[Info] {module_name}  min: {minimum.item()}  max: {maximum.item()}'
    else:
        module.check_gradient_parameter_range_msg = f'[Warning] {module_name}  there are not parameters in this layer'


def log_debug_message(path, layers, which_msg, which_epoch, which_batch):
    with open(path, 'a') as f:
        f.write(f'Epoch {which_epoch + 1} / Batch {which_batch + 1}:\n')
        for layer in layers:
            f.write(getattr(layer, which_msg))
            f.write('\n')
        f.write('\n')
