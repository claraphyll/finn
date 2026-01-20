# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import os
import shutil

import numpy as np
from qonnx.custom_op.general import im2col
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.deconvolutioninputgenerator import (
    DeconvolutionInputGenerator,
)
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

# RTL Deconvolution Input Generator / Sliding Window Generator (SWG)
# Matches and extends the functionality of all DeconvolutionInputGenerator_* functions
# in finn-hlslib by generating HDL code for two different implementation styles:
# - Addressable cyclic buffer: to be used when out_width <= in_width
# - Parallel registers + line buffers: to be used when out_width > in_width
# Supports non-square, 1D, strided, dilated, and depthwise Deconvolutions.
# Note: the actual data layout produced is different for depthwise and non-depthwise:
# * non-depthwise SWG: (1, OFMDim_H, OFMDim_W, K_H, K_W, IFMChannels/SIMD, SIMD)
# * depthwise SWG: (1, OFMDim_H, OFMDim_W, IFMChannels/SIMD, K_H, K_W, SIMD)

# NOTE: "Parallel" implementation style not yet implemented in this version!


class DeconvolutionInputGenerator_rtl(DeconvolutionInputGenerator, RTLBackend):
    """Class that corresponds to finn-rtllib swg module.
    Generates an RTL DeconvolutionInputGenerator implementation
    based on (System-)Verilog templates, defined in finn-rtllib/swg."""

    def __init__(self, onnx_node, **kwargs):
        print("Hello from rtl deconvinpgen")
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # additional parallelization parameter - not yet implemented
            "M": ("i", False, 1),
        }
        my_attrs.update(DeconvolutionInputGenerator.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_number_input_values(self):
        """Function to get the number of expected input values."""
        folded_ishape = self.get_folded_input_shape()
        num_input_elems = np.prod(folded_ishape[:-1])
        return num_input_elems

    def use_parallel_window_output(self):
        return self.get_nodeattr("parallel_window")

    def get_buffer_depth(self):
        """Returns total depth of the internal buffer, depending on
        implementation style."""
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        simd = self.get_nodeattr("SIMD")

        k_h, k_w = k
        h, w = ifm_dim
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        mmv_in = 1
        mmv_out = 1
        channel_factor = int(ifm_ch / simd)
        impl_style = self.select_impl_style()
        if impl_style == "default":
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
            ) * channel_factor
            # add additional buffer space in case of stride > 1
            # this minimizes cycle count as it allows an earlier pre-load of inputs
            buffer_depth = (
                buffer_min_size
                + max(
                    0,
                    ((stride_w - 1) - (int(mmv_out * k_h * k_w / mmv_in)))
                    * channel_factor,
                )
                + max(
                    0,
                    ((stride_h - 1) * w - (int(mmv_out * k_h * k_w / mmv_in)))
                    * channel_factor,
                )
            )
        elif impl_style == "parallel":
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w
            ) * channel_factor + 1
            buffer_depth = buffer_min_size + 1
        return buffer_depth

    def get_exp_cycles(self):
        impl_style = self.select_impl_style()

        if impl_style == "parallel":
            exp_cycles = self.get_number_input_values() + 2
        elif impl_style == "default":
            simd = self.get_nodeattr("SIMD")
            ifm_ch = self.get_nodeattr("IFMChannels")
            k = self.get_nodeattr("ConvKernelDim")
            ifm_dim = self.get_nodeattr("IFMDim")
            ofm_dim = self.get_nodeattr("OFMDim")
            stride = self.get_nodeattr("Stride")
            dilation = self.get_nodeattr("Dilation")
            depthwise = self.get_nodeattr("depthwise")
            ifm_dim_h, ifm_dim_w = ifm_dim
            ofm_dim_h, ofm_dim_w = ofm_dim
            k_h, k_w = k
            stride_h, stride_w = stride
            dilation_h, dilation_w = dilation

            channel_factor = int(ifm_ch / simd)
            if ifm_dim_h == 1 or ifm_dim_w == 1:
                # 1D case
                (
                    ifm_ch,
                    [ifm_dim_h, ifm_dim_w],
                    [ofm_dim_h, ofm_dim_w],
                    [k_h, k_w],
                    [stride_h, stride_w],
                    [dilation_h, dilation_w],
                ) = self.get_1d_conv_attrs_normalized()

                if depthwise:
                    exp_cycles = (
                        +ofm_dim_w * k_w * channel_factor
                        + channel_factor * (k_w - 1) * (stride_w - 1)
                        - (k_w - 1)
                        + 2
                    )
                else:
                    exp_cycles = ofm_dim_w * k_w * channel_factor + 2
            else:
                # 2D case
                buffer_min_size = (
                    (k_h - 1) * dilation_h * ifm_dim_w + (k_w - 1) * dilation_w + 1
                ) * channel_factor
                cycles_write_block = ofm_dim_w * k_w * k_h * channel_factor
                cycles_read_block = stride_w * ifm_dim_w * channel_factor
                max_cycles = max(cycles_write_block, cycles_read_block)
                if depthwise:
                    max_cycles += ofm_dim_w * (stride_w - 1) * (channel_factor - 1)
                exp_cycles = buffer_min_size + ofm_dim_h * max_cycles
                if depthwise:
                    exp_cycles += (stride_h - 1) * ifm_dim_w * channel_factor

        return int(exp_cycles)

    def bram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")
        impl_style = self.select_impl_style()
        [k_h, k_w] = self.get_nodeattr("ConvKernelDim")
        [ifm_dim_h, ifm_dim_w] = self.get_nodeattr("IFMDim")
        [dilation_h, dilation_w] = self.get_nodeattr("Dilation")

        if ram_style == "block" or ram_style == "auto":
            buffer_width = simd * self.get_input_datatype().bitwidth()
            if impl_style == "default":
                buffer_depth = self.get_buffer_depth()
                buffer_count = 1
            elif impl_style == "parallel":
                if ifm_dim_h == 1 or ifm_dim_w == 1:
                    return 0  # 1D case (no line buffers needed)
                kernel_width = (k_w - 1) * dilation_w + 1
                buffer_depth = (ifm_dim_w - kernel_width) + ifm_dim_w * (dilation_h - 1)
                buffer_count = k_h - 1

            # NOTE: Actual BRAM usage might be lower in some cases
            # due to imperfect modeling of Vivado behavior
            if buffer_depth <= 512:
                ram_width = 36
            elif buffer_depth <= 1024:
                ram_width = 18
            elif buffer_depth <= 2048:
                ram_width = 9
            elif buffer_depth <= 4096:
                ram_width = 4
            elif buffer_depth <= 8192:
                ram_width = 2
            else:
                ram_width = 1

            ram_cascade_depth = math.ceil(buffer_depth / 16384)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            cascade_savings = 0
            if buffer_depth > 16384:
                remainder_depth = buffer_depth % 16384
                if remainder_depth <= 512:
                    remainder_width = 36
                elif remainder_depth <= 1024:
                    remainder_width = 18
                elif remainder_depth <= 2048:
                    remainder_width = 9
                elif remainder_depth <= 4096:
                    remainder_width = 4
                elif remainder_depth <= 8192:
                    remainder_width = 2
                else:
                    remainder_width = 1

                remainder_cascade_width = math.ceil(buffer_width / remainder_width)
                cascade_savings = ram_cascade_width - remainder_cascade_width

            return int(
                (ram_cascade_depth * ram_cascade_width - cascade_savings) * buffer_count
            )
        else:
            return 0

    def lut_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")
        buffer_width = simd * self.get_input_datatype().bitwidth()
        buffer_depth = self.get_buffer_depth()
        if ram_style == "distributed":
            ram_luts = int(buffer_width * math.ceil(buffer_depth / 38))
        else:
            ram_luts = 0
        return 300 + ram_luts

    def uram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")
        impl_style = self.select_impl_style()
        [k_h, k_w] = self.get_nodeattr("ConvKernelDim")
        [ifm_dim_h, ifm_dim_w] = self.get_nodeattr("IFMDim")
        [dilation_h, dilation_w] = self.get_nodeattr("Dilation")

        if ram_style == "ultra":
            buffer_width = simd * self.get_input_datatype().bitwidth()
            if impl_style == "default":
                buffer_depth = self.get_buffer_depth()
                buffer_count = 1
            elif impl_style == "parallel":
                if ifm_dim_h == 1 or ifm_dim_w == 1:
                    return 0  # 1D case (no line buffers needed)
                kernel_width = (k_w - 1) * dilation_w + 1
                buffer_depth = (ifm_dim_w - kernel_width) + ifm_dim_w * (dilation_h - 1)
                buffer_count = k_h - 1

            ram_depth = 4096
            ram_width = 72
            ram_cascade_depth = math.ceil(buffer_depth / ram_depth)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            return int(ram_cascade_depth * ram_cascade_width * buffer_count)
        else:
            return 0

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")

        if mode == "cppsim":
            DeconvolutionInputGenerator.execute_node(self, context, graph)
            # if depthwise = 1
            # interleave channels such that cppsim of DeconvolutionInputGenerator_rtl
            # has a notion of SIMD parallelism. Subsequent VVAU_{hls/rtl} expects
            # the channels to be interleaved (i.e. to match their PE parallelism).
            if self.get_nodeattr("depthwise"):
                node = self.onnx_node
                im2col_out = context[node.output[0]]
                simd = getCustomOp(node).get_nodeattr("SIMD")
                ofm_h, ofm_w = getCustomOp(node).get_nodeattr("OFMDim")
                k_h, k_w = getCustomOp(node).get_nodeattr("ConvKernelDim")
                ifm_ch = getCustomOp(node).get_nodeattr("IFMChannels")
                im2col_out = im2col_out.reshape(
                    1, ofm_h, ofm_w, k_h * k_w, ifm_ch // simd, simd
                )
                im2col_out = im2col_out.transpose(0, 1, 2, 4, 3, 5)
                im2col_out = im2col_out.reshape(1, ofm_h, ofm_w, ifm_ch * k_h * k_w)
                context[node.output[0]] = im2col_out
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def prepare_codegen_default(self):
        """Fills code generation dict for the default implementation style by computing
        the incremental addressing scheme for the circular buffer."""
        if self.get_nodeattr("dynamic_mode"):
            raise NotImplementedError("Dynamic mode is not supported by deconv")
        else:
            template_select = "/finn-rtllib/deconv/src/"
        template_path = os.environ["FINN_ROOT"] + template_select
        code_gen_dict = {}

        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        depthwise = self.get_nodeattr("depthwise")
        simd = self.get_nodeattr("SIMD")

        k_h, k_w = k
        h, w = ifm_dim
        pad = [0, 0, 0, 0]  # padding happens in separate padding node for now
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        out_dim_h = im2col.compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = im2col.compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
        mmv_in = 1
        mmv_out = 1
        channel_factor = int(ifm_ch / simd)

        # compute minimal buffer length (assuming it holds 1 complete window)
        buffer_min_size = (
            (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
        ) * channel_factor

        buffer_actual_size = self.get_buffer_depth()
        code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

        # compute some intermediate values, e.g., kernel "width" = k_w incl. dilation
        # or cols/rows that are skipped due to imperfect stride<->dim combination
        kernel_width = (k_w - 1) * dilation_w + 1
        kernel_height = (k_h - 1) * dilation_h + 1
        skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
        skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

        code_gen_dict["$KERNEL_DIM_X$"] = [str(k_h)]
        code_gen_dict["$KERNEL_DIM_Y$"] = [str(k_w)]
        code_gen_dict["$INPUT_DIM_X$"] = [str(k_h)]
        code_gen_dict["$INPUT_DIM_Y$"] = [str(k_w)]

        code_gen_dict["$SIMD$"] = [str(simd)]

        return template_path, code_gen_dict

    def prepare_codegen_parallel(self):
        raise NotImplementedError(
            "Parallel implementation not supported for deconvolution input generator."
        )

    def select_impl_style(self):
        """Selects implementation style based on folding configuration."""
        simd = self.get_nodeattr("SIMD")
        M = self.get_nodeattr("M")
        depthwise = self.get_nodeattr("depthwise")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim_h, ifm_dim_w = ifm_dim
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        k_h, k_w = k
        kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
        kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

        # check for valid configuration
        assert (
            kernel_height <= ifm_dim_h
            and kernel_width <= ifm_dim_w
            and stride_h <= ifm_dim_h
            and stride_w <= ifm_dim_w
        ), "Illegal conv configuration: kernel or stride > FM dimension"

        # init folding config
        if self.get_nodeattr("parallel_window"):
            # mmv_in = M * 1
            mmv_out = M * k_h * k_w
        else:
            # mmv_in = 1
            mmv_out = 1
            assert ifm_ch % simd == 0, (
                "Constraint violated: SIMD must divide IFMChannels"
            )

        # choose implementation style
        if mmv_out > 1 or (k_h == 1 and k_w == 1):
            impl_style = "parallel"
            if depthwise or (k_h == 1 and k_w == 1):
                # allow SIMD < IFM_CH in depthwise mode (VVAU supports the resulting data layout)
                # also allowed for 1x1 kernel since depthwise and non-depthwise are equivalent
                assert ifm_ch % simd == 0, (
                    "Constraint violated: SIMD must divide IFMChannels"
                )
            else:
                assert ifm_ch == simd, (
                    "Constraint violated: SIMD must be equal to IFMChannels"
                )
        else:
            impl_style = "default"

        return impl_style

    def generate_hdl(self, model, fpgapart, clk):
        """Generates HDL code and wrapper for the IP, depending on required
        implementation style."""
        impl_style = self.select_impl_style()

        # prepare code generation by filling out dictionaries
        if impl_style == "default":
            template_path, code_gen_dict = self.prepare_codegen_default()
        else:
            raise NotImplementedError(
                "DeconvolutionInputGenerator only supports default impl style"
            )

        # add general parameters to dictionary
        code_gen_dict["$TOP_MODULE_NAME$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())
        code_gen_dict["$BIT_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]

        code_gen_dict["$IN_WIDTH_PADDED$"] = [
            str(roundup_to_integer_multiple(self.get_instream_width(), 8))
        ]
        code_gen_dict["$OUT_WIDTH_PADDED$"] = [
            str(roundup_to_integer_multiple(self.get_outstream_width(), 8))
        ]
        ram_style = self.get_nodeattr("ram_style")
        code_gen_dict["$RAM_STYLE$"] = ['"{}"'.format(ram_style)]

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        if self.get_nodeattr("dynamic_mode"):
            template_select = "/finn-rtllib/swg/swg_template_wrapper_dynamic.v"
        else:
            template_select = "/finn-rtllib/swg/swg_template_wrapper.v"
        with open(os.environ["FINN_ROOT"] + template_select, "r") as f:
            template_wrapper = f.read()
        with open(
            os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_template_axilite.v", "r"
        ) as f:
            template_axilite = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
            template_wrapper = template_wrapper.replace(key, code_gen_line)
            template_axilite = template_axilite.replace(key, code_gen_line)
        with open(
            os.path.join(
                code_gen_dir, self.get_nodeattr("gen_top_module") + "_impl.sv"
            ),
            "w",
        ) as f:
            f.write(template)
        with open(
            os.path.join(
                code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"
            ),
            "w",
        ) as f:
            f.write(template_wrapper)

        src_files = [
            "/finn-rtllib/deconv/src/kernal_buffer.sv"
            "/finn-rtllib/deconv/src/deconv_axi.sv"
            "/finn-rtllib/deconv/src/deconv.sv"
        ]
        # Copy static source file for common core components
        for file in src_files:
            shutil.copy2(os.environ["FINN_ROOT"] + file, code_gen_dir)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/deconv/src")
        else:
            code_gen_dir = ""
            rtllib_dir = ""
        verilog_files = [
            rtllib_dir + "swg_pkg.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + "_wrapper.v",
            code_gen_dir + self.get_nodeattr("gen_top_module") + "_impl.sv",
            rtllib_dir + "swg_common.sv",
        ]
        if self.get_nodeattr("dynamic_mode"):
            verilog_files.append(
                code_gen_dir + self.get_nodeattr("gen_top_module") + "_axilite.v"
            )

        return verilog_files

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "swg_pkg.sv",
            self.get_nodeattr("gen_top_module") + "_wrapper.v",
            self.get_nodeattr("gen_top_module") + "_impl.sv",
            "swg_common.sv",
        ]

        if self.get_nodeattr("dynamic_mode"):
            sourcefiles += [self.get_nodeattr("gen_top_module") + "_axilite.v"]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def get_verilog_top_module_intf_names(self):
        # Overload default HLSCustomOp implementation to add axilite control IF
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("dynamic_mode"):
            intf_names["axilite"] = ["s_axilite"]
        return intf_names
