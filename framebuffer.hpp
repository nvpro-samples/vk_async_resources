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


#pragma once

#include <nvvk/commands_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>

class FrameBuffer
{
public:
  nvvk::ResourceAllocator* m_memAllocator = nullptr;

  int      m_renderWidth  = 0;
  int      m_renderHeight = 0;
  uint32_t m_changeID     = 0;

  VkSampleCountFlagBits m_samplesUsed;
  VkFormat              m_colorFormat;
  VkFormat              m_depthStencilFormat;

  VkViewport m_viewport = {};
  VkRect2D   m_scissor  = {};

  VkRenderPass  m_passScene = VK_NULL_HANDLE;
  VkFramebuffer m_fboScene  = VK_NULL_HANDLE;

  VkRenderPass  m_passUI = VK_NULL_HANDLE;
  VkFramebuffer m_fboUI  = VK_NULL_HANDLE;

  nvvk::Image m_imgColor;
  nvvk::Image m_imgDepthStencil;

  VkImageView m_viewColor        = VK_NULL_HANDLE;
  VkImageView m_viewDepthStencil = VK_NULL_HANDLE;

  void init(nvvk::ResourceAllocator& memAllocator, VkFormat colorFormat)
  {
    VkResult result;

    VkDevice         device         = memAllocator.getDevice();
    VkPhysicalDevice physicalDevice = memAllocator.getPhysicalDevice();
    m_memAllocator                  = &memAllocator;

    m_samplesUsed = VK_SAMPLE_COUNT_1_BIT;

    // formats
    m_colorFormat        = colorFormat;
    m_depthStencilFormat = nvvk::findDepthStencilFormat(physicalDevice);

    // scene pass
    {
      VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

      // Create the render pass
      VkAttachmentDescription attachments[2] = {};
      attachments[0].format                  = m_colorFormat;
      attachments[0].samples                 = m_samplesUsed;
      attachments[0].loadOp                  = loadOp;
      attachments[0].storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
      attachments[0].initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachments[0].finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachments[0].flags                   = 0;

      attachments[1].format              = m_depthStencilFormat;
      attachments[1].samples             = m_samplesUsed;
      attachments[1].loadOp              = loadOp;
      attachments[1].storeOp             = VK_ATTACHMENT_STORE_OP_STORE;
      attachments[1].stencilLoadOp       = loadOp;
      attachments[1].stencilStoreOp      = VK_ATTACHMENT_STORE_OP_STORE;
      attachments[1].initialLayout       = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachments[1].finalLayout         = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachments[1].flags               = 0;
      VkSubpassDescription subpass       = {};
      subpass.pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.inputAttachmentCount       = 0;
      VkAttachmentReference colorRefs[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
      subpass.colorAttachmentCount       = NV_ARRAY_SIZE(colorRefs);
      subpass.pColorAttachments          = colorRefs;
      VkAttachmentReference depthRefs[1] = {{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
      subpass.pDepthStencilAttachment    = depthRefs;
      VkRenderPassCreateInfo rpInfo      = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
      rpInfo.attachmentCount             = NV_ARRAY_SIZE(attachments);
      rpInfo.pAttachments                = attachments;
      rpInfo.subpassCount                = 1;
      rpInfo.pSubpasses                  = &subpass;
      rpInfo.dependencyCount             = 0;

      result = vkCreateRenderPass(device, &rpInfo, nullptr, &m_passScene);
      NVVK_CHECK(result);
    }

    // ui pass
    {
      VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

      // Create the render pass
      VkAttachmentDescription attachments[1] = {};
      attachments[0].format                  = m_colorFormat;
      attachments[0].samples                 = m_samplesUsed;
      attachments[0].loadOp                  = loadOp;
      attachments[0].storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
      attachments[0].initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachments[0].finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachments[0].flags                   = 0;

      VkSubpassDescription subpass       = {};
      subpass.pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.inputAttachmentCount       = 0;
      VkAttachmentReference colorRefs[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
      subpass.colorAttachmentCount       = NV_ARRAY_SIZE(colorRefs);
      subpass.pColorAttachments          = colorRefs;
      subpass.pDepthStencilAttachment    = nullptr;
      VkRenderPassCreateInfo rpInfo      = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
      rpInfo.attachmentCount             = NV_ARRAY_SIZE(attachments);
      rpInfo.pAttachments                = attachments;
      rpInfo.subpassCount                = 1;
      rpInfo.pSubpasses                  = &subpass;
      rpInfo.dependencyCount             = 0;

      result = vkCreateRenderPass(device, &rpInfo, nullptr, &m_passUI);
      NVVK_CHECK(result);
    }
  }

  void updateResources(int width, int height)
  {
    VkResult result;
    VkDevice device = m_memAllocator->getDevice();

    m_changeID++;

    m_renderWidth  = width;
    m_renderHeight = height;

    if(m_imgColor.image != 0)
    {
      deinitResources();
    }

    // allocation
    {
      // color
      VkImageCreateInfo cbImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      cbImageInfo.imageType         = VK_IMAGE_TYPE_2D;
      cbImageInfo.format            = m_colorFormat;
      cbImageInfo.extent.width      = m_renderWidth;
      cbImageInfo.extent.height     = m_renderHeight;
      cbImageInfo.extent.depth      = 1;
      cbImageInfo.mipLevels         = 1;
      cbImageInfo.arrayLayers       = 1;
      cbImageInfo.samples           = m_samplesUsed;
      cbImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
      cbImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
      cbImageInfo.flags         = 0;
      cbImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

      // always enforce device local bit for framebuffer images!
      m_imgColor = m_memAllocator->createImage(cbImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
      dsImageInfo.format            = m_depthStencilFormat;
      dsImageInfo.extent.width      = m_renderWidth;
      dsImageInfo.extent.height     = m_renderHeight;
      dsImageInfo.extent.depth      = 1;
      dsImageInfo.mipLevels         = 1;
      dsImageInfo.arrayLayers       = 1;
      dsImageInfo.samples           = m_samplesUsed;
      dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
      dsImageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      dsImageInfo.flags             = 0;
      dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

      // always enforce device local bit for framebuffer images!
      m_imgDepthStencil = m_memAllocator->createImage(dsImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    {
      VkImageViewCreateInfo cbImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      cbImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
      cbImageViewInfo.format                          = m_colorFormat;
      cbImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
      cbImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
      cbImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
      cbImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
      cbImageViewInfo.flags                           = 0;
      cbImageViewInfo.subresourceRange.levelCount     = 1;
      cbImageViewInfo.subresourceRange.baseMipLevel   = 0;
      cbImageViewInfo.subresourceRange.layerCount     = 1;
      cbImageViewInfo.subresourceRange.baseArrayLayer = 0;
      cbImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

      cbImageViewInfo.image = m_imgColor.image;
      result                = vkCreateImageView(device, &cbImageViewInfo, nullptr, &m_viewColor);
      NVVK_CHECK(result);

      VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
      dsImageViewInfo.format                          = m_depthStencilFormat;
      dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
      dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
      dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
      dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
      dsImageViewInfo.flags                           = 0;
      dsImageViewInfo.subresourceRange.levelCount     = 1;
      dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
      dsImageViewInfo.subresourceRange.layerCount     = 1;
      dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
      dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

      dsImageViewInfo.image = m_imgDepthStencil.image;
      result                = vkCreateImageView(device, &dsImageViewInfo, nullptr, &m_viewDepthStencil);
      NVVK_CHECK(result);
    }

    {
      // Create scene framebuffer
      VkImageView bindInfos[2];
      bindInfos[0] = m_viewColor;
      bindInfos[1] = m_viewDepthStencil;

      VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
      fbInfo.pAttachments            = bindInfos;
      fbInfo.width                   = m_renderWidth;
      fbInfo.height                  = m_renderHeight;
      fbInfo.layers                  = 1;
      fbInfo.renderPass              = m_passScene;

      result = vkCreateFramebuffer(device, &fbInfo, nullptr, &m_fboScene);
      NVVK_CHECK(result);
    }

    {
      // Create ui framebuffer
      VkImageView bindInfos[1];
      bindInfos[0] = m_viewColor;

      VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
      fbInfo.pAttachments            = bindInfos;
      fbInfo.width                   = m_renderWidth;
      fbInfo.height                  = m_renderHeight;
      fbInfo.layers                  = 1;
      fbInfo.renderPass              = m_passUI;

      result = vkCreateFramebuffer(device, &fbInfo, nullptr, &m_fboUI);
      NVVK_CHECK(result);
    }

    {
      m_viewport.x        = 0;
      m_viewport.y        = 0;
      m_viewport.width    = float(m_renderWidth);
      m_viewport.height   = float(m_renderHeight);
      m_viewport.minDepth = 0.0f;
      m_viewport.maxDepth = 1.0f;

      m_scissor.offset.x      = 0;
      m_scissor.offset.y      = 0;
      m_scissor.extent.width  = m_renderWidth;
      m_scissor.extent.height = m_renderHeight;
    }
  }

  void cmdUpdateBarriers(VkCommandBuffer cmd)
  {
    VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(0);
    VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                                   | nvvk::makeAccessMaskPipelineStageFlags(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkImageMemoryBarrier memBarriers[] = {
        nvvk::makeImageMemoryBarrier(m_imgColor.image, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
        nvvk::makeImageMemoryBarrier(m_imgDepthStencil.image, 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT),
    };

    vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, NULL, 0, NULL, NV_ARRAY_SIZE(memBarriers), memBarriers);
  }


  inline void reverseImageMemoryBarrier(VkImageMemoryBarrier& barrier)
  {
    std::swap(barrier.oldLayout, barrier.newLayout);
    std::swap(barrier.srcAccessMask, barrier.dstAccessMask);
    std::swap(barrier.dstQueueFamilyIndex, barrier.srcQueueFamilyIndex);
  }

  void cmdBlitToSwapChain(VkCommandBuffer cmd, const nvvk::SwapChain& swapChain)
  {
    // transition from render/present to blit
    VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                                   | nvvk::makeAccessMaskPipelineStageFlags(0);

    VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(VK_ACCESS_TRANSFER_READ_BIT)
                                   | nvvk::makeAccessMaskPipelineStageFlags(VK_ACCESS_TRANSFER_WRITE_BIT);

    VkImageMemoryBarrier memBarriers[] = {
        nvvk::makeImageMemoryBarrier(m_imgColor.image, 0, VK_ACCESS_TRANSFER_READ_BIT,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        nvvk::makeImageMemoryBarrier(swapChain.getActiveImage(), 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, NULL, 0, NULL, NV_ARRAY_SIZE(memBarriers), memBarriers);


    {
      // blit to vk backbuffer
      VkImageBlit region               = {0};
      region.srcOffsets[1].x           = m_renderWidth;
      region.srcOffsets[1].y           = m_renderHeight;
      region.srcOffsets[1].z           = 1;
      region.dstOffsets[1].x           = swapChain.getWidth();
      region.dstOffsets[1].y           = swapChain.getHeight();
      region.dstOffsets[1].z           = 1;
      region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.dstSubresource.layerCount = 1;
      region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.srcSubresource.layerCount = 1;

      vkCmdBlitImage(cmd, m_imgColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapChain.getActiveImage(),
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    }

    // transition back to render/present

    std::swap(srcPipe, dstPipe);
    reverseImageMemoryBarrier(memBarriers[0]);
    reverseImageMemoryBarrier(memBarriers[1]);

    vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, NULL, 0, NULL, NV_ARRAY_SIZE(memBarriers), memBarriers);
  }

  void deinitResources()
  {
    VkDevice device = m_memAllocator->getDevice();

    vkDestroyImageView(device, m_viewColor, nullptr);
    vkDestroyImageView(device, m_viewDepthStencil, nullptr);    
    vkDestroyFramebuffer(device, m_fboScene, nullptr);
    vkDestroyFramebuffer(device, m_fboUI, nullptr);
    m_memAllocator->destroy(m_imgColor);
    m_memAllocator->destroy(m_imgDepthStencil);
  }

  void deinit()
  {
    VkDevice device = m_memAllocator->getDevice();

    vkDestroyRenderPass(device, m_passScene, nullptr);
    vkDestroyRenderPass(device, m_passUI, nullptr);

    deinitResources();
  }
};
