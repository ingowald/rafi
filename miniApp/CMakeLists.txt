# Copyright 2025 Ingo Wald
# SPDX-License-Identifier: Apache-2.0

# all optix device functions have to be in a separate .cu file that
# will then get compiled to PTX and loaded into optix in precompiled
# PTX form at runtime. use this helper macro to compile this device
# code into PTX, and embed it into a binary array in a helper libray
# that we can then link to our final app (so app has this PTX code in
# binary form without having to explicitly load it from a text file at
# runtime).
#
embed_ptx(
  OUTPUT_TARGET      embedded_devCode
  PTX_LINK_LIBRARIES rafi owl::owl
  SOURCES            devCode.cu
)

# host- and pure CUDA (i.e., non-optix device programs) side of our
# miniapp
add_executable(miniApp
  miniApp.h
  miniApp.cu
)

# linking the mini-app with our helper tool for face iteration
# (`faceIteration`, from <root>/faceIteration/`), for the GPU side MPI
# compositing (`<root>/compositing/`), and our embedded optix device
# code (`embedded_devCode`)
target_link_libraries(miniApp PUBLIC
  embedded_devCode
  rafi
  )
 
