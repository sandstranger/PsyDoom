import os
import subprocess
import sys

files_to_compile = [
    [ "colored.frag",                   "compiled/SPIRV_colored_frag.bin.h",                    "frag", "gSPIRV_colored_frag"                   ],
    [ "colored.vert",                   "compiled/SPIRV_colored_vert.bin.h",                    "vert", "gSPIRV_colored_vert"                   ],
    [ "crossfade.frag",                 "compiled/SPIRV_crossfade_frag.bin.h",                  "frag", "gSPIRV_crossfade_frag"                 ],
    [ "crossfade_gamma.frag",           "compiled/SPIRV_crossfade_gamma_frag.bin.h",            "frag", "gSPIRV_crossfade_gamma_frag"           ],
    [ "gamma_adjust_blit.frag",         "compiled/SPIRV_gamma_adjust_blit_frag.bin.h",          "frag", "gSPIRV_gamma_adjust_blit_frag"         ],
    [ "gamma_adjust_post_process.frag", "compiled/SPIRV_gamma_adjust_post_process_frag.bin.h",  "frag", "gSPIRV_gamma_adjust_post_process_frag" ],
    [ "msaa_resolve.frag",              "compiled/SPIRV_msaa_resolve_frag.bin.h",               "frag", "gSPIRV_msaa_resolve_frag"              ],
    [ "ndc_position_only.vert",         "compiled/SPIRV_ndc_position_only_vert.bin.h",          "vert", "gSPIRV_ndc_position_only_vert"         ],
    [ "ndc_textured.frag",              "compiled/SPIRV_ndc_textured_frag.bin.h",               "frag", "gSPIRV_ndc_textured_frag"              ],
    [ "ndc_textured.vert",              "compiled/SPIRV_ndc_textured_vert.bin.h",               "vert", "gSPIRV_ndc_textured_vert"              ],
    [ "sky.frag",                       "compiled/SPIRV_sky_frag.bin.h",                        "frag", "gSPIRV_sky_frag"                       ],
    [ "sky.vert",                       "compiled/SPIRV_sky_vert.bin.h",                        "vert", "gSPIRV_sky_vert"                       ],
    [ "ui.vert",                        "compiled/SPIRV_ui_vert.bin.h",                         "vert", "gSPIRV_ui_vert"                        ],
    [ "ui_16bpp.frag",                  "compiled/SPIRV_ui_16bpp_frag.bin.h",                   "frag", "gSPIRV_ui_16bpp_frag"                  ],
    [ "ui_4bpp.frag",                   "compiled/SPIRV_ui_4bpp_frag.bin.h",                    "frag", "gSPIRV_ui_4bpp_frag"                   ],
    [ "ui_8bpp.frag",                   "compiled/SPIRV_ui_8bpp_frag.bin.h",                    "frag", "gSPIRV_ui_8bpp_frag"                   ],
    [ "world.frag",                     "compiled/SPIRV_world_frag.bin.h",                      "frag", "gSPIRV_world_frag"                     ],
    [ "world.vert",                     "compiled/SPIRV_world_vert.bin.h",                      "vert", "gSPIRV_world_vert"                     ],
]

def compile_shader(input_glsl_file, output_c_file, shader_stage, c_var_name):
    temp_spv = output_c_file + ".spv"
    opt_spv = output_c_file + ".opt.spv"

    try:
        result = subprocess.call([
            "glslc", "-I", ".", "--target-env=vulkan1.0",
            "-fshader-stage=" + shader_stage,
            "-O", "-o", temp_spv, input_glsl_file
        ])
        if result != 0:
            raise RuntimeError(f"Compile FAILED for file: {input_glsl_file}")

        result = subprocess.call([
            "spirv-opt",
            "--eliminate-dead-code-aggressive",
            "--inline-entry-points-exhaustive",
            "--flatten-decorations",
            "--scalar-replacement",
            "--eliminate-local-multi-store",
            "--merge-blocks",
            "--strip-debug",
            temp_spv,
            "-o", opt_spv
        ])
        if result != 0:
            raise RuntimeError(f"Optimization FAILED for file: {input_glsl_file}")

        with open(opt_spv, "rb") as f:
            data = f.read()

        with open(output_c_file, "w") as f:
            f.write(f"static const uint32_t {c_var_name}[] = {{\n")
            for i in range(0, len(data), 4):
                word = data[i:i+4]
                while len(word) < 4:
                    word += b'\x00'
                value = int.from_bytes(word, byteorder='little')
                f.write(f"0x{value:08x},\n")
            f.write("};\n")

    finally:
        for file in (temp_spv, opt_spv):
            if os.path.exists(file):
                os.remove(file)

def main():
    for job_spec in files_to_compile:
        compile_shader(job_spec[0], job_spec[1], job_spec[2], job_spec[3])

main()