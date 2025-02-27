import unittest
import numpy as np
from streamlined_transpiler import LLVMToCudaTranspiler
import llvmlite.binding as llvm

class TestLLVMToCudaTranspiler(unittest.TestCase):

    def setUp(self):
        self.transpiler = LLVMToCudaTranspiler()

    def test_initialization(self):
        self.assertIsInstance(self.transpiler, LLVMToCudaTranspiler)
        self.assertEqual(self.transpiler.target_triple, "nvptx64-nvidia-cuda")
        self.assertEqual(self.transpiler.gpu_arch, "sm_70")

    def test_parse_module_valid_ir(self):
        llvm_ir_code = """
        define i32 @add(i32 %a, i32 %b) {
          %result = add i32 %a, %b
          ret i32 %result
        }
        """
        mod = self.transpiler.parse_module(llvm_ir_code)
        self.assertIsInstance(mod, llvm.ModuleRef)

    def test_parse_module_invalid_ir(self):
        llvm_ir_code = """
        This is not valid LLVM IR
        """
        with self.assertRaises(llvm.ParseError):
            self.transpiler.parse_module(llvm_ir_code)

    def test_llvm_to_ptx_simple_kernel(self):
        llvm_ir_code = """
        define void @kernel.add(float* nocapture %out, float* nocapture %in1, float* nocapture %in2) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          %in1.ptr = getelementptr float, float* %in1, i32 %tid
          %in2.ptr = getelementptr float, float* %in2, i32 %tid
          %out.ptr = getelementptr float, float* %out, i32 %tid
          %val1 = load float, float* %in1.ptr
          %val2 = load float, float* %in2.ptr
          %add = fadd float %val1, %val2
          store float %add, float* %out.ptr
          ret void
        }

        declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

        attributes #0 = { nounwind }
        """
        ptx_code = self.transpiler.llvm_to_ptx(llvm_ir_code)
        self.assertIsInstance(ptx_code, str)
        self.assertTrue("%kernel.add" in ptx_code)

    def test_extract_kernels_single_kernel(self):
        llvm_ir_code = """
        define void @kernel.add(float* nocapture %out, float* nocapture %in1, float* nocapture %in2) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          ret void
        }

        define i32 @host_function(i32 %a) {
          ret i32 %a
        }
        """
        kernels = self.transpiler.extract_kernels(llvm_ir_code)
        self.assertIn("kernel.add", kernels)
        self.assertNotIn("host_function", kernels)

    def test_transpile_to_cuda_simple_kernel(self):
        llvm_ir_code = """
        define void @kernel.add(float* nocapture %out, float* nocapture %in1, float* nocapture %in2) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          %in1.ptr = getelementptr float, float* %in1, i32 %tid
          %val1 = load float, float* %in1.ptr
          ret void
        }

        declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
        """
        cuda_code = self.transpiler.transpile_to_cuda(llvm_ir_code)
        self.assertIsInstance(cuda_code, str)
        self.assertTrue("__global__ void kernel_add" in cuda_code)
        self.assertTrue("/* Unsupported: call */" in cuda_code) # llvm.nvvm.read.ptx.sreg.tid.x is not directly transpilable

    def test_convert_llvm_type(self):
        test_types = [
            ("i32", "int"),
            ("float*", "float*"),
            ("[4 x float]", "float4"),
            ("void", "void"),
            ("i1", "bool")
        ]
        for llvm_type_str, expected_cuda_type in test_types:
            llvm_type = llvm.Type.parse(llvm_type_str)
            cuda_type = self.transpiler._convert_llvm_type(llvm_type)
            self.assertEqual(cuda_type, expected_cuda_type)

    def test_convert_instruction_binary_op(self):
        llvm_ir_code = """
        define i32 @test_func(i32 %a, i32 %b) {
          %add_result = add i32 %a, %b
          ret i32 %add_result
        }
        """
        mod = self.transpiler.parse_module(llvm_ir_code)
        func = mod.functions[0]
        add_instr = func.blocks[0].instructions[0] # Get the add instruction
        cuda_instr = self.transpiler._convert_instruction(add_instr)
        self.assertEqual(cuda_instr.strip(), "add_result = a + b;")

    def test_convert_instruction_ret(self):
        llvm_ir_code = """
        define i32 @test_func(i32 %a) {
          ret i32 %a
        }
        """
        mod = self.transpiler.parse_module(llvm_ir_code)
        func = mod.functions[0]
        ret_instr = func.blocks[0].instructions[0]
        cuda_instr = self.transpiler._convert_instruction(ret_instr)
        self.assertEqual(cuda_instr.strip(), "return a;")

    def test_end_to_end_execute_simple_add(self):
        llvm_ir_code = """
        define i32 @kernel_add(i32 %a) {
          %result = add i32 %a, 10
          ret i32 %result
        }
        """
        input_data = 5
        result, _ = self.transpiler.end_to_end_execute(llvm_ir_code, input_data=input_data)
        self.assertEqual(result, 15)

    def test_end_to_end_execute_array_add(self):
        llvm_ir_code = """
        define void @kernel_array_add(float* %out, float* %in) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          %in.ptr = getelementptr float, float* %in, i32 %tid
          %out.ptr = getelementptr float, float* %out, i32 %tid
          %val = load float, float* %in.ptr
          %add = fadd float %val, 1.0
          store float %add, float* %out.ptr
          ret void
        }
        declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
        """
        input_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result, _ = self.transpiler.end_to_end_execute(llvm_ir_code, input_data=input_array, output_type=np.float32)
        expected_output = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected_output)

    def test_execute_cpu_for_validation_simple_add(self):
        llvm_ir_code = """
        define i32 @kernel_add(i32 %a) {
          %result = add i32 %a, 10
          ret i32 %result
        }
        """
        input_data = 5
        result_cpu = self.transpiler.execute_cpu_for_validation(llvm_ir_code, input_data=input_data)
        self.assertEqual(result_cpu, 15)

    def test_compute_dominators_simple_cfg(self):
        llvm_ir_code = """
        define void @func() {
        entry:
          br label %block1

        block1:                                           ; preds = %entry
          br label %block2

        block2:                                           ; preds = %block1
          br label %block3

        block3:                                           ; preds = %block2
          ret void
        }
        """
        dom_sets, block_names = self.transpiler.compute_dominators(llvm_ir_code)
        expected_dom_sets = {
            "entry": {"entry"},
            "block1": {"entry", "block1"},
            "block2": {"entry", "block1", "block2"},
            "block3": {"entry", "block1", "block2", "block3"}
        }
        self.assertEqual(dom_sets, expected_dom_sets)

    def test_compute_dominators_if_else_cfg(self):
        llvm_ir_code = """
        define void @func(i1 %cond) {
        entry:
          br i1 %cond, label %then_block, label %else_block

        then_block:                                       ; preds = %entry
          br label %merge_block

        else_block:                                       ; preds = %entry
          br label %merge_block

        merge_block:                                      ; preds = %then_block, %else_block
          ret void
        }
        """
        dom_sets, block_names = self.transpiler.compute_dominators(llvm_ir_code)
        expected_dom_sets = {
            "entry": {"entry"},
            "then_block": {"entry", "then_block"},
            "else_block": {"entry", "else_block"},
            "merge_block": {"entry", "merge_block"} # entry dominates merge in this simplified version
        }
        self.assertEqual(dom_sets, expected_dom_sets)

    def test_execute_kernel_compilation_failure(self):
        invalid_ptx_code = "This is not valid PTX code"
        function_name = "invalid_kernel"
        with self.assertRaises(Exception) as context:
            self.transpiler.compile_and_load_ptx(invalid_ptx_code, function_name)
        self.assertIn("Failed to compile PTX", str(context.exception))

    def test_end_to_end_execution_kernel_not_found(self):
        llvm_ir_code_no_kernel = """
        define i32 @host_function(i32 %a) {
          ret i32 %a
        }
        """
        with self.assertRaises(ValueError) as context:
            self.transpiler.end_to_end_execute(llvm_ir_code_no_kernel)
        self.assertIn("No kernel function found", str(context.exception))

if __name__ == '__main__':
    unittest.main()
