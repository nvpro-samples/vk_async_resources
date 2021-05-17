/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef _COMMON_H_
#define _COMMON_H_

/////////////////////////////////////////////////

#define DSET_SCENE              0
#define DSET_SCENE_UBO_VIEW     0

#define VERTEX_POS    0
#define VERTEX_NORMAL 1
#define VERTEX_TEX    2

#ifdef __cplusplus
namespace glsl {
  using namespace nvmath;
#endif

struct ViewData {
  mat4  viewProjMatrix;
  mat4  viewProjMatrixI;
  mat4  viewMatrix;
  mat4  viewMatrixIT;

  vec4  viewPos;
  vec4  viewDir;
  
  ivec2 viewport;
  vec2  viewportf;
};

#ifdef __cplusplus
}
#endif

#endif
