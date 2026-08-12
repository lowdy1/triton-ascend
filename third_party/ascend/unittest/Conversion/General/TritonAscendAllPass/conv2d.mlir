// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @triton_conv2d_3d_kernel(%input_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %weight_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_in_offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %c_in_offsets_0 = tt.expand_dims %c_in_offsets {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c_in_offsets_1 = tt.expand_dims %c_in_offsets_0 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %h_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %h_offsets_2 = tt.expand_dims %h_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %h_offsets_3 = tt.expand_dims %h_offsets_2 {axis = 2 : i32} : tensor<1x16xi32> -> tensor<1x16x1xi32>
    %w_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %w_offsets_4 = tt.expand_dims %w_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %w_offsets_5 = tt.expand_dims %w_offsets_4 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %input_offs = arith.constant dense<256> : tensor<8x1x1xi32>
    %input_offs_6 = arith.constant dense<16> : tensor<1x16x1xi32>
    %input_offs_h = arith.muli %c_in_offsets_1, %input_offs : tensor<8x1x1xi32>
    %input_offs_hw = arith.muli %h_offsets_3, %input_offs_6 : tensor<1x16x1xi32>
    %input_offs_t0 = tt.broadcast %input_offs_h : tensor<8x1x1xi32> -> tensor<8x16x1xi32>
    %input_offs_t1 = tt.broadcast %input_offs_hw : tensor<1x16x1xi32> -> tensor<8x16x1xi32>
    %input_offs_t2 = arith.addi %input_offs_t0, %input_offs_t1 : tensor<8x16x1xi32>
    %input_offs_t3 = tt.broadcast %input_offs_t2 : tensor<8x16x1xi32> -> tensor<8x16x16xi32>
    %input_offs_t4 = tt.broadcast %w_offsets_5 : tensor<1x1x16xi32> -> tensor<8x16x16xi32>
    %input_offs_t5 = arith.addi %input_offs_t3, %input_offs_t4 : tensor<8x16x16xi32>
    %input_tensor = tt.splat %input_ptr : !tt.ptr<f16> -> tensor<8x16x16x!tt.ptr<f16>>
    %input_tensor_7 = tt.addptr %input_tensor, %input_offs_t5 : tensor<8x16x16x!tt.ptr<f16>>, tensor<8x16x16xi32>
    %input_tile = tt.load %input_tensor_7 : tensor<8x16x16x!tt.ptr<f16>>
    %co_offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %co_offs_0 = tt.expand_dims %co_offs {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %co_offs_1 = tt.expand_dims %co_offs_0 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %co_offs_2 = tt.expand_dims %co_offs_1 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %ci_offs = tt.expand_dims %c_in_offsets {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %ci_offs_0 = tt.expand_dims %ci_offs {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %ci_offs_1 = tt.expand_dims %ci_offs_0 {axis = 3 : i32} : tensor<1x8x1xi32> -> tensor<1x8x1x1xi32>
    %kh_offs = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %kh_offs_0 = tt.expand_dims %kh_offs {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %kh_offs_1 = tt.expand_dims %kh_offs_0 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %kh_offs_2 = tt.expand_dims %kh_offs_1 {axis = 3 : i32} : tensor<1x1x3xi32> -> tensor<1x1x3x1xi32>
    %kw_offs = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %kw_offs_0 = tt.expand_dims %kw_offs {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %kw_offs_1 = tt.expand_dims %kw_offs_0 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %weight_offs_c = arith.constant dense<72> : tensor<4x1x1x1xi32>
    %weight_offs_ci = arith.constant dense<9> : tensor<1x8x1x1xi32>
    %weight_offs_kh = arith.constant dense<3> : tensor<1x1x3x1xi32>
    %weight_t0 = arith.muli %co_offs_2, %weight_offs_c : tensor<4x1x1x1xi32>
    %weight_t1 = arith.muli %ci_offs_1, %weight_offs_ci : tensor<1x8x1x1xi32>
    %weight_t2 = arith.muli %kh_offs_2, %weight_offs_kh : tensor<1x1x3x1xi32>
    %weight_t3 = tt.broadcast %weight_t0 : tensor<4x1x1x1xi32> -> tensor<4x8x1x1xi32>
    %weight_t4 = tt.broadcast %weight_t1 : tensor<1x8x1x1xi32> -> tensor<4x8x1x1xi32>
    %weight_t5 = arith.addi %weight_t3, %weight_t4 : tensor<4x8x1x1xi32>
    %weight_t6 = tt.broadcast %weight_t5 : tensor<4x8x1x1xi32> -> tensor<4x8x3x1xi32>
    %weight_t7 = tt.broadcast %weight_t2 : tensor<1x1x3x1xi32> -> tensor<4x8x3x1xi32>
    %weight_t8 = arith.addi %weight_t6, %weight_t7 : tensor<4x8x3x1xi32>
    %weight_t9 = tt.broadcast %weight_t8 : tensor<4x8x3x1xi32> -> tensor<4x8x3x3xi32>
    %weight_t10 = tt.broadcast %kw_offs_1 : tensor<1x1x3xi32> -> tensor<4x8x3x3xi32>
    %weight_offs = arith.addi %weight_t9, %weight_t10 : tensor<4x8x3x3xi32>
    %weight_tensor = tt.splat %weight_ptr : !tt.ptr<f16> -> tensor<4x8x3x3x!tt.ptr<f16>>
    %weight_tensor_ = tt.addptr %weight_tensor, %weight_offs : tensor<4x8x3x3x!tt.ptr<f16>>, tensor<4x8x3x3xi32>
    %weight_tile = tt.load %weight_tensor_ : tensor<4x8x3x3x!tt.ptr<f16>>
    %output = ascend.conv2d(%input_tile, %weight_tile) {dilation = array<i32: 1, 1>, groups = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>} : (tensor<8x16x16xf16>, tensor<4x8x3x3xf16>) -> tensor<4x14x14xf16>
    %lo_h = tt.make_range {end = 14 : i32, start = 0 : i32} : tensor<14xi32>
    %lo_h_0 = tt.expand_dims %lo_h {axis = 0 : i32} : tensor<14xi32> -> tensor<1x14xi32>
    %lo_h_1 = tt.expand_dims %lo_h_0 {axis = 2 : i32} : tensor<1x14xi32> -> tensor<1x14x1xi32>
    %lo_w = tt.make_range {end = 14 : i32, start = 0 : i32} : tensor<14xi32>
    %lo_w_0 = tt.expand_dims %lo_w {axis = 0 : i32} : tensor<14xi32> -> tensor<1x14xi32>
    %lo_w_1 = tt.expand_dims %lo_w_0 {axis = 1 : i32} : tensor<1x14xi32> -> tensor<1x1x14xi32>
    %out_offs_c = arith.constant dense<196> : tensor<4x1x1xi32>
    %out_offs_h = arith.constant dense<14> : tensor<1x14x1xi32>
    %co_offs_3 = tt.expand_dims %co_offs {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %co_offs_4 = tt.expand_dims %co_offs_3 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %out_t0 = arith.muli %co_offs_4, %out_offs_c : tensor<4x1x1xi32>
    %out_t1 = arith.muli %lo_h_1, %out_offs_h : tensor<1x14x1xi32>
    %out_t2 = tt.broadcast %out_t0 : tensor<4x1x1xi32> -> tensor<4x14x1xi32>
    %out_t3 = tt.broadcast %out_t1 : tensor<1x14x1xi32> -> tensor<4x14x1xi32>
    %out_t4 = arith.addi %out_t2, %out_t3 : tensor<4x14x1xi32>
    %out_t5 = tt.broadcast %out_t4 : tensor<4x14x1xi32> -> tensor<4x14x14xi32>
    %out_t6 = tt.broadcast %lo_w_1 : tensor<1x1x14xi32> -> tensor<4x14x14xi32>
    %out_offs = arith.addi %out_t5, %out_t6 : tensor<4x14x14xi32>
    %0 = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<4x14x14x!tt.ptr<f16>>
    %1 = tt.addptr %0, %out_offs : tensor<4x14x14x!tt.ptr<f16>>, tensor<4x14x14xi32>
    tt.store %1, %output : tensor<4x14x14x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_conv2d_3d_kernel(
// CHECK: %[[VAL_0:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [8, 16, 16], strides: [256, 16, 1]
// CHECK: %[[VAL_1:.*]] = memref.alloc() : memref<8x16x16xf16>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [4, 8, 3, 3], strides: [72, 9, 3, 1]
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<4x8x3x3xf16>
// CHECK: memref.copy %[[VAL_2]], %[[VAL_3]]
// CHECK: hfusion.conv2d
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: groups = 1 : i32
// CHECK-SAME: padding = array<i64: 0, 0>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK: %[[VAL_6:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [4, 14, 14], strides: [196, 14, 1]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[VAL_6]]

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @triton_conv2d_4d_kernel(%input_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %weight_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // Input offsets: (N=2, C=8, H=16, W=16), row-major strides [2048, 256, 16, 1]
    %n_offsets = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %n_offsets_0 = tt.expand_dims %n_offsets {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %n_offsets_1 = tt.expand_dims %n_offsets_0 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %n_offsets_2 = tt.expand_dims %n_offsets_1 {axis = 3 : i32} : tensor<2x1x1xi32> -> tensor<2x1x1x1xi32>
    %c_in_offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %c_in_offsets_0 = tt.expand_dims %c_in_offsets {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %c_in_offsets_1 = tt.expand_dims %c_in_offsets_0 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %c_in_offsets_2 = tt.expand_dims %c_in_offsets_1 {axis = 3 : i32} : tensor<1x8x1xi32> -> tensor<1x8x1x1xi32>
    %h_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %h_offsets_0 = tt.expand_dims %h_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %h_offsets_1 = tt.expand_dims %h_offsets_0 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %h_offsets_2 = tt.expand_dims %h_offsets_1 {axis = 3 : i32} : tensor<1x1x16xi32> -> tensor<1x1x16x1xi32>
    %w_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %w_offsets_0 = tt.expand_dims %w_offsets {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %w_offsets_1 = tt.expand_dims %w_offsets_0 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %w_offsets_2 = tt.expand_dims %w_offsets_1 {axis = 2 : i32} : tensor<1x1x16xi32> -> tensor<1x1x1x16xi32>
    %input_offs_n = arith.constant dense<2048> : tensor<2x1x1x1xi32>
    %input_offs_c = arith.constant dense<256> : tensor<1x8x1x1xi32>
    %input_offs_h = arith.constant dense<16> : tensor<1x1x16x1xi32>
    %input_t0 = arith.muli %n_offsets_2, %input_offs_n : tensor<2x1x1x1xi32>
    %input_t1 = arith.muli %c_in_offsets_2, %input_offs_c : tensor<1x8x1x1xi32>
    %input_t2 = arith.muli %h_offsets_2, %input_offs_h : tensor<1x1x16x1xi32>
    %input_t3 = tt.broadcast %input_t0 : tensor<2x1x1x1xi32> -> tensor<2x8x1x1xi32>
    %input_t4 = tt.broadcast %input_t1 : tensor<1x8x1x1xi32> -> tensor<2x8x1x1xi32>
    %input_t5 = arith.addi %input_t3, %input_t4 : tensor<2x8x1x1xi32>
    %input_t6 = tt.broadcast %input_t5 : tensor<2x8x1x1xi32> -> tensor<2x8x16x1xi32>
    %input_t7 = tt.broadcast %input_t2 : tensor<1x1x16x1xi32> -> tensor<2x8x16x1xi32>
    %input_t8 = arith.addi %input_t6, %input_t7 : tensor<2x8x16x1xi32>
    %input_t9 = tt.broadcast %input_t8 : tensor<2x8x16x1xi32> -> tensor<2x8x16x16xi32>
    %input_t10 = tt.broadcast %w_offsets_2 : tensor<1x1x1x16xi32> -> tensor<2x8x16x16xi32>
    %input_offsets = arith.addi %input_t9, %input_t10 : tensor<2x8x16x16xi32>
    %input_tensor = tt.splat %input_ptr : !tt.ptr<f16> -> tensor<2x8x16x16x!tt.ptr<f16>>
    %input_tensor_ = tt.addptr %input_tensor, %input_offsets : tensor<2x8x16x16x!tt.ptr<f16>>, tensor<2x8x16x16xi32>
    %input_tile = tt.load %input_tensor_ : tensor<2x8x16x16x!tt.ptr<f16>>
    // Weight offsets: (C_out=4, C_in=8, kH=3, kW=3), strides [72, 9, 3, 1]
    %co_offsets = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %co_offsets_0 = tt.expand_dims %co_offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %co_offsets_1 = tt.expand_dims %co_offsets_0 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %co_offsets_2 = tt.expand_dims %co_offsets_1 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %ci_offsets = tt.expand_dims %c_in_offsets {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %ci_offsets_0 = tt.expand_dims %ci_offsets {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %ci_offsets_1 = tt.expand_dims %ci_offsets_0 {axis = 3 : i32} : tensor<1x8x1xi32> -> tensor<1x8x1x1xi32>
    %kh_offsets = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %kh_offsets_0 = tt.expand_dims %kh_offsets {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %kh_offsets_1 = tt.expand_dims %kh_offsets_0 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %kh_offsets_2 = tt.expand_dims %kh_offsets_1 {axis = 3 : i32} : tensor<1x1x3xi32> -> tensor<1x1x3x1xi32>
    %kw_offsets = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
    %kw_offsets_0 = tt.expand_dims %kw_offsets {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
    %kw_offsets_1 = tt.expand_dims %kw_offsets_0 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %weight_offs_co = arith.constant dense<72> : tensor<4x1x1x1xi32>
    %weight_offs_ci = arith.constant dense<9> : tensor<1x8x1x1xi32>
    %weight_offs_kh = arith.constant dense<3> : tensor<1x1x3x1xi32>
    %weight_t0 = arith.muli %co_offsets_2, %weight_offs_co : tensor<4x1x1x1xi32>
    %weight_t1 = arith.muli %ci_offsets_1, %weight_offs_ci : tensor<1x8x1x1xi32>
    %weight_t2 = arith.muli %kh_offsets_2, %weight_offs_kh : tensor<1x1x3x1xi32>
    %weight_t3 = tt.broadcast %weight_t0 : tensor<4x1x1x1xi32> -> tensor<4x8x1x1xi32>
    %weight_t4 = tt.broadcast %weight_t1 : tensor<1x8x1x1xi32> -> tensor<4x8x1x1xi32>
    %weight_t5 = arith.addi %weight_t3, %weight_t4 : tensor<4x8x1x1xi32>
    %weight_t6 = tt.broadcast %weight_t5 : tensor<4x8x1x1xi32> -> tensor<4x8x3x1xi32>
    %weight_t7 = tt.broadcast %weight_t2 : tensor<1x1x3x1xi32> -> tensor<4x8x3x1xi32>
    %weight_t8 = arith.addi %weight_t6, %weight_t7 : tensor<4x8x3x1xi32>
    %weight_t9 = tt.broadcast %weight_t8 : tensor<4x8x3x1xi32> -> tensor<4x8x3x3xi32>
    %weight_t10 = tt.broadcast %kw_offsets_1 : tensor<1x1x3xi32> -> tensor<4x8x3x3xi32>
    %weight_offsets = arith.addi %weight_t9, %weight_t10 : tensor<4x8x3x3xi32>
    %weight_tensor = tt.splat %weight_ptr : !tt.ptr<f16> -> tensor<4x8x3x3x!tt.ptr<f16>>
    %weight_tensor_ = tt.addptr %weight_tensor, %weight_offsets : tensor<4x8x3x3x!tt.ptr<f16>>, tensor<4x8x3x3xi32>
    %weight_tile = tt.load %weight_tensor_ : tensor<4x8x3x3x!tt.ptr<f16>>
    // Asymmetric padding [pad_top=1, pad_bottom=0, pad_left=2, pad_right=0]:
    // H_out = (16 + 1 + 0 - 2 - 1) + 1 = 14, W_out = (16 + 2 + 0 - 2 - 1) + 1 = 16
    %output = ascend.conv2d(%input_tile, %weight_tile) {dilation = array<i32: 1, 1>, groups = 1 : i64, padding = array<i32: 1, 0, 2, 0>, stride = array<i32: 1, 1>} : (tensor<2x8x16x16xf16>, tensor<4x8x3x3xf16>) -> tensor<2x4x14x16xf16>
    // Output offsets: (N=2, C_out=4, H_out=14, W_out=16), strides [896, 224, 16, 1]
    %lo_h = tt.make_range {end = 14 : i32, start = 0 : i32} : tensor<14xi32>
    %lo_h_0 = tt.expand_dims %lo_h {axis = 0 : i32} : tensor<14xi32> -> tensor<1x14xi32>
    %lo_h_1 = tt.expand_dims %lo_h_0 {axis = 1 : i32} : tensor<1x14xi32> -> tensor<1x1x14xi32>
    %lo_h_2 = tt.expand_dims %lo_h_1 {axis = 3 : i32} : tensor<1x1x14xi32> -> tensor<1x1x14x1xi32>
    %lo_w = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %lo_w_0 = tt.expand_dims %lo_w {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %lo_w_1 = tt.expand_dims %lo_w_0 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %lo_w_2 = tt.expand_dims %lo_w_1 {axis = 2 : i32} : tensor<1x1x16xi32> -> tensor<1x1x1x16xi32>
    %out_offs_n = arith.constant dense<896> : tensor<2x1x1x1xi32>
    %out_offs_c = arith.constant dense<224> : tensor<1x4x1x1xi32>
    %out_offs_h = arith.constant dense<16> : tensor<1x1x14x1xi32>
    %co_offsets_3 = tt.expand_dims %co_offsets {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %co_offsets_4 = tt.expand_dims %co_offsets_3 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
    %co_offsets_5 = tt.expand_dims %co_offsets_4 {axis = 3 : i32} : tensor<1x4x1xi32> -> tensor<1x4x1x1xi32>
    %out_t0 = arith.muli %n_offsets_2, %out_offs_n : tensor<2x1x1x1xi32>
    %out_t1 = arith.muli %co_offsets_5, %out_offs_c : tensor<1x4x1x1xi32>
    %out_t2 = arith.muli %lo_h_2, %out_offs_h : tensor<1x1x14x1xi32>
    %out_t3 = tt.broadcast %out_t0 : tensor<2x1x1x1xi32> -> tensor<2x4x1x1xi32>
    %out_t4 = tt.broadcast %out_t1 : tensor<1x4x1x1xi32> -> tensor<2x4x1x1xi32>
    %out_t5 = arith.addi %out_t3, %out_t4 : tensor<2x4x1x1xi32>
    %out_t6 = tt.broadcast %out_t5 : tensor<2x4x1x1xi32> -> tensor<2x4x14x1xi32>
    %out_t7 = tt.broadcast %out_t2 : tensor<1x1x14x1xi32> -> tensor<2x4x14x1xi32>
    %out_t8 = arith.addi %out_t6, %out_t7 : tensor<2x4x14x1xi32>
    %out_t9 = tt.broadcast %out_t8 : tensor<2x4x14x1xi32> -> tensor<2x4x14x16xi32>
    %out_t10 = tt.broadcast %lo_w_2 : tensor<1x1x1x16xi32> -> tensor<2x4x14x16xi32>
    %output_offsets = arith.addi %out_t9, %out_t10 : tensor<2x4x14x16xi32>
    %0 = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<2x4x14x16x!tt.ptr<f16>>
    %1 = tt.addptr %0, %output_offsets : tensor<2x4x14x16x!tt.ptr<f16>>, tensor<2x4x14x16xi32>
    tt.store %1, %output : tensor<2x4x14x16x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_conv2d_4d_kernel(
// CHECK: %[[VAL_0:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [2, 8, 16, 16], strides: [2048, 256, 16, 1]
// CHECK: %[[VAL_1:.*]] = memref.alloc() : memref<2x8x16x16xf16>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [4, 8, 3, 3], strides: [72, 9, 3, 1]
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<4x8x3x3xf16>
// CHECK: memref.copy %[[VAL_2]], %[[VAL_3]]
// CHECK: hfusion.conv2d
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: groups = 1 : i32
// CHECK-SAME: padding = array<i64: 1, 2>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK: %[[VAL_6:.*]] = memref.reinterpret_cast
// CHECK-SAME: to offset: [0], sizes: [2, 4, 14, 16], strides: [896, 224, 16, 1]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[VAL_6]]
