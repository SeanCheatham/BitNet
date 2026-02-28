"""Shared code generation utilities for TL1 and TL2 codegen."""

import os
from configparser import ConfigParser


def gen_transform_code(kernel_shapes):
    """Generate C++ ggml_bitnet_transform_tensor function.

    This function is identical between TL1 and TL2 codegen.
    """
    kernel_code = "\n\
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor, float scale) {\n\
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {\n\
        return;\n\
    }\n\
\n\
    int k = tensor->ne[0];\n\
    int m = tensor->ne[1];\n\
    const int lut_scales_size = 1;\n\
    int bk = 0;\n\
    int bm = 0;\n"

    kernel_code = "".join([kernel_code, "\n\
    if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[0][0], kernel_shapes[0][1])])

    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "else if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[i][0], kernel_shapes[i][1])])

    kernel_code = "".join([kernel_code, "\n\
    const int n_tile_num = m / bm;\n\
    const int BK = bk;\n\
    uint8_t * qweights;\n\
    bitnet_float_type * scales;\n\
\n\
    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));\n\
    qweights = (uint8_t *) tensor->data;\n\
    scales[0] = (bitnet_float_type) scale;\n\
\n\
    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;\n\
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {\n\
        /* .lut_scales_size = */ lut_scales_size,\n\
        /* .BK              = */ BK,\n\
        /* .n_tile_num      = */ n_tile_num,\n\
        /* .qweights        = */ qweights,\n\
        /* .scales          = */ scales\n\
    };\n\
}\n"])

    return kernel_code


def write_kernel_config(kernel_shapes, BM_list, BK_list, bm_list, output_dir):
    """Write kernel_config.ini file with kernel shape parameters."""
    config = ConfigParser()
    for i in range(len(kernel_shapes)):
        section = f'Kernels_{i}'
        config.add_section(section)
        config.set(section, 'M', str(kernel_shapes[i][0]))
        config.set(section, 'K', str(kernel_shapes[i][1]))
        config.set(section, 'BM', str(BM_list[i]))
        config.set(section, 'BK', str(BK_list[i]))
        config.set(section, 'bmm', str(bm_list[i]))

    with open(os.path.join(output_dir, "kernel_config.ini"), 'w') as configfile:
        config.write(configfile)
