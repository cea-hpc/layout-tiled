// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef LAYOUT_TILED_IMPL_HOST_DEVICE
#if defined(LAYOUT_TILED_IMPL_HAS_CUDA) || defined(LAYOUT_TILED_IMPL_HAS_HIP)
#define LAYOUT_TILED_IMPL_HOST_DEVICE __host__ __device__
#else
#define LAYOUT_TILED_IMPL_HOST_DEVICE
#endif
#endif

#ifndef LAYOUT_TILED_FORCE_INLINE_FUNCTION
#define LAYOUT_TILED_FORCE_INLINE_FUNCTION __attribute__((always_inline)) LAYOUT_TILED_IMPL_HOST_DEVICE
#endif

#ifndef LAYOUT_TILED_INLINE_FUNCTION
#define LAYOUT_TILED_INLINE_FUNCTION inline LAYOUT_TILED_IMPL_HOST_DEVICE
#endif
