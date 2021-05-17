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


#version 440

#extension GL_GOOGLE_include_directive : enable
#include "common.h"

layout(location=0) in Interpolants {
  vec3 pos;
  vec3 normal;
  vec2 tex;
} IN;

layout(location=0,index=0) out vec4 out_Color;

void main()
{
  vec3  light = normalize(vec3(0,2,-2));
  float intensity = abs(dot(normalize(IN.normal),light));
  vec4  color = vec4(0.75) + vec4( fract(IN.tex*32), 1, 1) * 0.25;
  
  out_Color = color * intensity;
}
