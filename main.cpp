/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <random>
#include <string>

#include <nvpwindow.hpp>

#include <nvh/geometry.hpp>
#include <nvh/misc.hpp>

#include <nvvk/allocator_dma_vk.hpp>

#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/extensions_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/swapchain_vk.hpp>

#include <nvmath/nvmath.h>
#include <nvmath/nvmath_glsltypes.h>

#include <imgui/imgui_helper.h>
#include <imgui/imgui_impl_vk.h>

#include "framebuffer.hpp"

#include "common.h"

/*
  # vk_async_resources

  This sample showcases several framework helper classes to aid
  development with Vulkan using its C api.

  The test particularly demonstrates asynchronous resource creation
  and transfers.

  You can reload shaders with "R" key or pressing the button.
*/


class Sample
{
public:
  // ---------------------------------------------------------
  // shortcuts
  VkDevice         m_device              = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice      = VK_NULL_HANDLE;
  VkQueue          m_queue               = VK_NULL_HANDLE;
  uint32_t         m_queueFamily         = ~0;
  VkQueue          m_queueTransfer       = VK_NULL_HANDLE;
  uint32_t         m_queueTransferFamily = ~0;
  // ---------------------------------------------------------
  // generic utilities
  nvvk::ProfilerVK          m_profilerVK;
  nvvk::ShaderModuleManager m_shaderManager;

  nvvk::BatchSubmission m_submission;
  nvvk::RingFences      m_ringFences;
  nvvk::RingCommandPool m_ringCmdPool;

  nvvk::DeviceMemoryAllocator m_memAllocator;

  // the framebuffer class is not totally generic, but good for simple work
  // should create your own class for different samples as pass setups etc. will
  // vary
  FrameBuffer m_frameBuffer;

  uint32_t m_frame  = 0;
  double   m_uiTime = 0;

  // -- Test ---------------------------------------------------------

  struct AsyncTransferJob
  {
    VkCommandBuffer              cmd         = VK_NULL_HANDLE;
    uint32_t                     frameSignal = 0;
    VkSemaphore                  semaphore   = VK_NULL_HANDLE;
    VkFence                      fence       = VK_NULL_HANDLE;
    bool                         print       = true;
    std::vector<nvvk::BufferDma> purgeableResources;
  };

  struct Test
  {
    VkPipeline                   pipeline = VK_NULL_HANDLE;
    nvvk::DescriptorSetContainer container;
    nvvk::ShaderModuleID         moduleVS;
    nvvk::ShaderModuleID         moduleFS;

    nvvk::BufferDma viewUbo;
    nvvk::BufferDma geoVbo;
    nvvk::BufferDma geoIbo;

    nvvk::AllocatorDma allocatorDma;

    std::vector<VkDrawIndexedIndirectCommand> drawCmds;

    AsyncTransferJob  transfer;
    nvvk::CommandPool transferCmdPool;
    VkFence           transferFence;
    VkSemaphore       transferSemaphore;

    // ui/tweakable
    bool useAsync        = true;
    bool useRegeneration = false;
  };

  Test m_test;


  bool init(nvvk::Context& context, uint32_t width, uint32_t height)
  {
    // generic infrastructure classes
    m_device              = context.m_device;
    m_physicalDevice      = context.m_physicalDevice;
    m_queue               = context.m_queueGCT.queue;
    m_queueFamily         = context.m_queueGCT.familyIndex;
    m_queueTransfer       = context.m_queueT.queue;
    m_queueTransferFamily = context.m_queueT.familyIndex;

    m_ringFences.init(m_device);
    m_ringCmdPool.init(m_device, m_queueFamily, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

    m_submission.init(m_queue);

    m_shaderManager.init(m_device);
    m_shaderManager.m_filetype = nvh::ShaderFileManager::FILETYPE_GLSL;

    m_shaderManager.addDirectory(std::string(PROJECT_NAME));
    m_shaderManager.addDirectory(std::string("GLSL_" PROJECT_NAME));
    m_shaderManager.addDirectory(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY));

    m_profilerVK.init(m_device, m_physicalDevice);
    m_profilerVK.setLabelUsage(context.hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME));

    // primary memory allocator used
    // in this simple case we use small chunks, however for real-world we recommend larger sizes

    m_memAllocator.init(m_device, m_physicalDevice, 16 * 1024 * 1024);
    m_frameBuffer.init(m_memAllocator, VK_FORMAT_R8G8B8A8_UNORM);
    updateFrameBuffer(width, height);


    ImGuiH::Init(width, height, this);
    ImGui::InitVK(context.m_device, context.m_physicalDevice, context.m_queueGCT, context.m_queueGCT.familyIndex,
                  m_frameBuffer.m_passUI);

    return initTest();
  }

  void deinit()
  {
    // Guard by synchronization since it is unsafe to delete some objects while in use
    vkDeviceWaitIdle(m_device);

    deinitTest();

    ImGui::ShutdownVK();

    m_frameBuffer.deinit();
    m_memAllocator.deinit();
    m_ringFences.deinit();
    m_ringCmdPool.deinit();

    // Delete all accumulated shader modules
    m_shaderManager.deinit();

    m_profilerVK.deinit();
  }

  //////////////////////////////////////////////////////////////////////////

  bool initTest()
  {
    // internal subsystems

    // simplified allocator wrapper
    m_test.allocatorDma.init(m_device, m_physicalDevice, &m_memAllocator, 16 * 1024 * 1024);
    // in this particular sample we want to keep staging memory around,
    // as we keep re-using it
    m_test.allocatorDma.getStaging()->setFreeUnusedOnRelease(false);

    // command pool for async transfers
    m_test.transferCmdPool.init(m_device, m_queueTransferFamily);

    VkSemaphoreCreateInfo semInfo = nvvk::make<VkSemaphoreCreateInfo>();
    vkCreateSemaphore(m_device, &semInfo, nullptr, &m_test.transferSemaphore);

    VkFenceCreateInfo fenceInfo = nvvk::make<VkFenceCreateInfo>();
    vkCreateFence(m_device, &fenceInfo, nullptr, &m_test.transferFence);

    // geometry
    initTestGeometry(1);

    // scene descriptors
    m_test.viewUbo = m_test.allocatorDma.createBuffer(sizeof(glsl::ViewData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    {
      {
        // define our descriptorset and pipelinelayouts

        VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        m_test.container.addBinding(DSET_SCENE_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);

        m_test.container.init(m_device);
        m_test.container.initLayout();
        m_test.container.initPipeLayout(0);
        m_test.container.initPool(1);
      }
      {
        // update the descriptorset

        std::vector<VkWriteDescriptorSet> descrWrites;
        VkDescriptorBufferInfo            viewInfo = {m_test.viewUbo.buffer, 0, sizeof(glsl::ViewData)};
        descrWrites.push_back(m_test.container.makeWrite(0, DSET_SCENE_UBO_VIEW, &viewInfo));

        vkUpdateDescriptorSets(m_device, uint32_t(descrWrites.size()), descrWrites.data(), 0, nullptr);
      }

      m_shaderManager.registerInclude("common.h", "common.h");

      m_test.moduleVS = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "test.vert.glsl");
      assert(m_shaderManager.isValid(m_test.moduleVS));
      m_test.moduleFS = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "test.frag.glsl");
      assert(m_shaderManager.isValid(m_test.moduleFS));

      // we use a dedicated function to enable hot-reloading of shaders
      initTestPipeline();
    }

    return true;
  }

  typedef nvh::geometry::Vertex Vertex;

  void initTestGeometry(uint32_t subdiv)
  {
    nvh::geometry::Torus<Vertex> torus(subdiv * 32, subdiv * 32);

    VkDeviceSize vboSize = torus.getVerticesSize();
    VkDeviceSize iboSize = torus.getTriangleIndicesSize();
    const void*  vboData = (const void*)torus.m_vertices.data();
    const void*  iboData = (const void*)torus.m_indicesTriangles.data();

    m_test.drawCmds.clear();

    VkDrawIndexedIndirectCommand drawCmd = {0};
    drawCmd.indexCount                   = torus.getTriangleIndicesCount();
    drawCmd.instanceCount                = 1024 / subdiv;  // to create some reasonable load
    m_test.drawCmds.push_back(drawCmd);

    if(m_test.useAsync)
    {
      // use simplified AllocatorDma wrapper class for resource creation and staging upload

      AsyncTransferJob& job = m_test.transfer;

      // assign semaphore to signal drawing to only start if upload completed
      job.semaphore = m_test.transferSemaphore;

      // assign fence to signal host to recycle memory later
      job.fence = m_test.transferFence;
      vkResetFences(m_device, 1, &job.fence);

      // We only have one transfer job here, so we use the same fence/semaphore
      // normally you would need some pooling system. Or purely base everything
      // upon submission ticks/frame counters etc.

      // keep record in which frame we got triggered
      job.frameSignal = m_frame;

      // get command buffer for staging operations
      job.cmd = m_test.transferCmdPool.createCommandBuffer();
      {
        auto timeOnce = m_profilerVK.timeSingle("Upload", job.cmd, true);
        m_test.geoVbo = m_test.allocatorDma.createBuffer(job.cmd, vboSize, vboData, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        m_test.geoIbo = m_test.allocatorDma.createBuffer(job.cmd, iboSize, iboData, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      }

      vkEndCommandBuffer(job.cmd);

      // finalize the staging job for later cleanup of resources
      // associates all current staging resources with the fence
      m_test.allocatorDma.finalizeStaging(job.fence);

      // submit staged transfers
      VkSubmitInfo submitInfo = nvvk::makeSubmitInfo(1, &job.cmd, 1, &job.semaphore);
      vkQueueSubmit(m_queueTransfer, 1, &submitInfo, job.fence);

      // next graphics submission must wait for transfer completion
      m_submission.enqueueWait(job.semaphore, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    else
    {
      // scope class usage on regular graphics queue
      // WARNING this is blocking the device, slow

      nvvk::StagingMemoryManager staging(m_device, m_physicalDevice);
      {
        nvvk::ScopeCommandBuffer cmd(m_device, m_queueFamily, m_queue);
        auto                     timeOnce = m_profilerVK.timeSingle("Upload", cmd);

        // showcases individual subsystem usage, not using AllocatorDMA
        m_test.geoVbo.buffer = m_memAllocator.createBuffer(vboSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, m_test.geoVbo.allocation);
        m_test.geoIbo.buffer = m_memAllocator.createBuffer(iboSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, m_test.geoIbo.allocation);

        staging.cmdToBuffer(cmd, m_test.geoVbo.buffer, 0, vboSize, vboData);
        staging.cmdToBuffer(cmd, m_test.geoIbo.buffer, 0, iboSize, iboData);

        // no need to "finalize or release" for staging
        // since we release all intermediate resources at staging destructor anyway
      }
    }
  }

  void initTestPipeline()
  {
    if(m_test.pipeline)
    {
      vkDestroyPipeline(m_device, m_test.pipeline, nullptr);
      m_test.pipeline = VK_NULL_HANDLE;
    }

    nvvk::GraphicsPipelineState     gfxState;
    nvvk::GraphicsPipelineGenerator gfxGen(m_device, m_test.container.getPipeLayout(), m_frameBuffer.m_passScene, gfxState);
    gfxState.depthStencilState.depthTestEnable  = true;
    gfxState.depthStencilState.depthWriteEnable = true;
    gfxState.depthStencilState.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    gfxState.rasterizationState.cullMode  = VK_CULL_MODE_BACK_BIT;
    gfxState.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    gfxState.inputAssemblyState.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    gfxState.addAttributeDescription(nvvk::GraphicsPipelineState::makeVertexInputAttribute(VERTEX_POS, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                                                                           offsetof(Vertex, position)));
    gfxState.addAttributeDescription(nvvk::GraphicsPipelineState::makeVertexInputAttribute(VERTEX_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                                                                           offsetof(Vertex, normal)));
    gfxState.addAttributeDescription(nvvk::GraphicsPipelineState::makeVertexInputAttribute(VERTEX_TEX, 0, VK_FORMAT_R32G32_SFLOAT,
                                                                                           offsetof(Vertex, texcoord)));
    gfxState.addBindingDescription(nvvk::GraphicsPipelineState::makeVertexInputBinding(0, sizeof(Vertex)));
    gfxGen.addShader(m_shaderManager.get(m_test.moduleVS), VK_SHADER_STAGE_VERTEX_BIT);
    gfxGen.addShader(m_shaderManager.get(m_test.moduleFS), VK_SHADER_STAGE_FRAGMENT_BIT);
    gfxState.addDynamicStateEnable(VK_DYNAMIC_STATE_VIEWPORT);
    gfxState.addDynamicStateEnable(VK_DYNAMIC_STATE_SCISSOR);

    m_test.pipeline = gfxGen.createPipeline();
    assert(m_test.pipeline != VK_NULL_HANDLE);
  }

  void deinitTest()
  {
    m_test.allocatorDma.destroy(m_test.viewUbo);
    m_test.allocatorDma.destroy(m_test.geoVbo);
    m_test.allocatorDma.destroy(m_test.geoIbo);

    vkDestroyPipeline(m_device, m_test.pipeline, nullptr);

    if(m_test.transfer.cmd)
    {
      m_test.transferCmdPool.destroy(m_test.transfer.cmd);
      m_test.transfer.cmd = nullptr;
    }

    deleteAsyncJobResources(m_test.transfer);

    m_test.transferCmdPool.deinit();
    m_test.container.deinit();
    m_test.allocatorDma.deinit();

    vkDestroyFence(m_device, m_test.transferFence, nullptr);
    vkDestroySemaphore(m_device, m_test.transferSemaphore, nullptr);
  }

  //////////////////////////////////////////////////////////////////////////

  void drawTest(VkCommandBuffer cmd)
  {

    {
      glsl::ViewData viewData = {};
      viewData.viewport       = nvmath::ivec2(m_frameBuffer.m_renderWidth, m_frameBuffer.m_renderHeight);
      viewData.viewportf      = nvmath::vec2(m_frameBuffer.m_renderWidth, m_frameBuffer.m_renderHeight);

      nvmath::mat4 projection =
          nvmath::perspectiveVK(60.0f, float(m_frameBuffer.m_renderWidth) / float(m_frameBuffer.m_renderHeight), 0.000001f, 10.0f);
      nvmath::mat4 view = nvmath::look_at(nvmath::vec3(0.0, 1.5, -1.5), nvmath::vec3(0.0, 0.0, 0.0), nvmath::vec3(0, 1, 0));
      nvmath::mat4 viewI = nvmath::invert(view);

      viewData.viewProjMatrix  = projection * view;
      viewData.viewProjMatrixI = nvmath::invert(viewData.viewProjMatrix);
      viewData.viewMatrix      = view;
      viewData.viewMatrixIT    = nvmath::transpose(viewI);

      viewData.viewPos = viewData.viewMatrixIT.row(3);
      viewData.viewDir = -view.row(2);

      vkCmdUpdateBuffer(cmd, m_test.viewUbo.buffer, 0, sizeof(glsl::ViewData), &viewData);
    }

    {
      VkRenderPassBeginInfo renderPassBeginInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      renderPassBeginInfo.renderPass               = m_frameBuffer.m_passScene;
      renderPassBeginInfo.framebuffer              = m_frameBuffer.m_fboScene;
      renderPassBeginInfo.renderArea.offset.x      = 0;
      renderPassBeginInfo.renderArea.offset.y      = 0;
      renderPassBeginInfo.renderArea.extent.width  = m_frameBuffer.m_renderWidth;
      renderPassBeginInfo.renderArea.extent.height = m_frameBuffer.m_renderHeight;
      renderPassBeginInfo.clearValueCount          = 2;

      nvmath::vec4f bgColor(0.2, 0.2, 0.2, 0.0);

      VkClearValue clearValues[2];
      clearValues[0].color.float32[0]     = bgColor.x;
      clearValues[0].color.float32[1]     = bgColor.y;
      clearValues[0].color.float32[2]     = bgColor.z;
      clearValues[0].color.float32[3]     = bgColor.w;
      clearValues[1].depthStencil.depth   = 1.0f;
      clearValues[1].depthStencil.stencil = 0;
      renderPassBeginInfo.pClearValues    = clearValues;

      vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      vkCmdSetViewport(cmd, 0, 1, &m_frameBuffer.m_viewport);
      vkCmdSetScissor(cmd, 0, 1, &m_frameBuffer.m_scissor);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_test.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_test.container.getPipeLayout(), 0, 1,
                            m_test.container.getSets(), 0, nullptr);

    VkDeviceSize vboOffsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_test.geoVbo.buffer, vboOffsets);
    vkCmdBindIndexBuffer(cmd, m_test.geoIbo.buffer, 0, VK_INDEX_TYPE_UINT32);
    for(auto& draw : m_test.drawCmds)
    {
      vkCmdDrawIndexed(cmd, draw.indexCount, draw.instanceCount, draw.firstIndex, draw.vertexOffset, draw.firstInstance);
    }

    vkCmdEndRenderPass(cmd);
  }

  void drawUI(VkCommandBuffer cmd, const ImDrawData* imguiDrawData)
  {
    VkRenderPassBeginInfo renderPassBeginInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPassBeginInfo.renderPass               = m_frameBuffer.m_passUI;
    renderPassBeginInfo.framebuffer              = m_frameBuffer.m_fboUI;
    renderPassBeginInfo.renderArea.offset.x      = 0;
    renderPassBeginInfo.renderArea.offset.y      = 0;
    renderPassBeginInfo.renderArea.extent.width  = m_frameBuffer.m_renderWidth;
    renderPassBeginInfo.renderArea.extent.height = m_frameBuffer.m_renderHeight;
    renderPassBeginInfo.clearValueCount          = 0;
    renderPassBeginInfo.pClearValues             = nullptr;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdSetViewport(cmd, 0, 1, &m_frameBuffer.m_viewport);
    vkCmdSetScissor(cmd, 0, 1, &m_frameBuffer.m_scissor);

    ImGui::RenderDrawDataVK(cmd, imguiDrawData);

    vkCmdEndRenderPass(cmd);
  }

  //////////////////////////////////////////////////////////////////////////

  void updateFrameBuffer(uint32_t width, uint32_t height)
  {
    m_frameBuffer.updateResources(width, height);
    {
      nvvk::ScopeCommandBuffer cmd(m_device, m_queueFamily, m_queue);
      m_frameBuffer.cmdUpdateBarriers(cmd);
    }
  }

  void reloadShaders()
  {
    VkResult result = vkQueueWaitIdle(m_queue);
    if(nvvk::checkResult(result))
    {
      exit(-1);
    }

    m_shaderManager.reloadShaderModules();
    initTestPipeline();
  }

  void deleteAsyncJobResources(AsyncTransferJob& job)
  {
    for(auto& itbuffer : job.purgeableResources)
    {
      m_test.allocatorDma.destroy(itbuffer);
    }
    job.purgeableResources.clear();
  }

  void tryCleanupAsyncJob(AsyncTransferJob& job)
  {
    // staging will directly test the fence we gave it for this transfer job
    // and release resources
    m_test.allocatorDma.releaseStaging();

    // we also check if fence was triggered, that means the copy has completed
    if(job.fence && vkGetFenceStatus(m_device, job.fence) == VK_SUCCESS)
    {
      // free used cmdbuffer
      m_test.transferCmdPool.destroy(job.cmd);
      job.cmd   = VK_NULL_HANDLE;
      job.fence = VK_NULL_HANDLE;
    }
    // wait a few frames until we know that the frame waiting for the semaphore
    // has completed (the fence above only tells us the copy operation has completed,
    // not whether the queue waiting for this copy had progressed)
    if(job.semaphore && m_frame > job.frameSignal + nvvk::DEFAULT_RING_SIZE)
    {
      job.semaphore = VK_NULL_HANDLE;

      // delete unused resources
      deleteAsyncJobResources(job);
    }
  }

  void processFrame(nvvk::SwapChain* swapChain)
  {
    m_profilerVK.beginFrame();

    ImGui::NewFrame();
    processUI(m_frameBuffer.m_renderWidth, m_frameBuffer.m_renderHeight, m_profilerVK.getMicroSeconds() * 0.000001);

    // dynamically recreate geometry to showcase async behaviour
    // the overall frametime will be faster with async true

    uint32_t recreateCycle = nvvk::DEFAULT_RING_SIZE + 2;
    if(m_test.useRegeneration && m_frame && m_frame % recreateCycle == 0)
    {

      // push old resources for deletion
      m_test.transfer.purgeableResources.push_back(m_test.geoIbo);
      m_test.transfer.purgeableResources.push_back(m_test.geoVbo);

      initTestGeometry(1 + ((m_frame / 4) % 2));

      if(!m_test.useAsync)
      {
        // delete directly due to sync'ed behavior
        deleteAsyncJobResources(m_test.transfer);
      }
    }

    // Ensure the host cannot race too far in front of the device.
    // This way we never run outside nvvk::MAX_RING_FRAMES
    //
    // The longer our cycle count, the more latency we potentially introduce.
    // if the host is much faster at processing frames than the device.

    m_ringFences.setCycleAndWait(m_frame);
    m_ringCmdPool.setCycle(m_frame);

    // Pick up a new command buffer every frame and
    // record our principle operations

    VkCommandBuffer cmd =
        m_ringCmdPool.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    {
      // draw test
      {
        auto scopeTimer = m_profilerVK.timeRecurring("Draw", cmd);
        drawTest(cmd);
      }
      // draw ui
      {
        auto scopeTimer = m_profilerVK.timeRecurring("UI", cmd);
        ImGui::Render();
        drawUI(cmd, ImGui::GetDrawData());
      }
      // blit to swapchain
      if(swapChain)
      {
        auto scopeTimer = m_profilerVK.timeRecurring("Blit", cmd);
        m_frameBuffer.cmdBlitToSwapChain(cmd, *swapChain);
      }
    }
    vkEndCommandBuffer(cmd);

    // Submit command buffer to the graphics (and presentation) queue.
    // In this sample only one cmd buffer is submitted, but you could imagine more complex setups
    // where you might want to batch multiple submissions. In general the amount of total VkQueueSubmits
    // should be low (single digits).

    if(swapChain)
    {
      m_submission.enqueueWait(swapChain->getActiveReadSemaphore(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
      m_submission.enqueueSignal(swapChain->getActiveWrittenSemaphore());
    }

    m_submission.enqueue(cmd);
    m_submission.execute(m_ringFences.getFence());
    m_profilerVK.endFrame();

    if(m_test.useAsync)
    {
      tryCleanupAsyncJob(m_test.transfer);
    }


    // print some stats
    if(m_profilerVK.getTotalFrames() % 64 == 0)
    {
      std::string stats;
      m_profilerVK.print(stats);
      printf("%s\n", stats.c_str());

      VkDeviceSize allocSize;
      VkDeviceSize usedSize;
      float        util;
      util = m_memAllocator.getUtilization(allocSize, usedSize);
      printf("Memory:  %7d / %7d KB\n", uint32_t(allocSize / 1024), uint32_t(usedSize / 1024));
      util = m_test.allocatorDma.getStaging()->getUtilization(allocSize, usedSize);
      printf("Staging: %7d / %7d KB\n", uint32_t(allocSize / 1024), uint32_t(usedSize / 1024));
    }

    if(m_test.transfer.print)
    {
      nvh::Profiler::TimerInfo info;
      if(m_profilerVK.getTimerInfo("Upload", info))
      {
        LOGI("Upload Time: GPU %6d CPU %6d\n", uint32_t(info.gpu.average), uint32_t(info.cpu.average));
        m_test.transfer.print = false;
      }
    }

    ImGui::EndFrame();

    m_frame++;
  }

  void processUI(int width, int height, double time)
  {
    // Update imgui configuration
    auto& imgui_io       = ImGui::GetIO();
    imgui_io.DeltaTime   = static_cast<float>(time - m_uiTime);
    imgui_io.DisplaySize = ImVec2(width, height);

    m_uiTime = time;

    ImGui::SetNextWindowPos(ImVec2(5, 5));
    ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
    if(ImGui::Begin("NVIDIA " PROJECT_NAME, nullptr))
    {
      ImGui::PushItemWidth(120);
      ImGui::Checkbox("use async transfer", &m_test.useAsync);
      ImGui::Checkbox("dynamic scene generation ", &m_test.useRegeneration);
      ImGui::Text("    (flickers a bit)");
      ImGui::Separator();
      {
        nvh::Profiler::TimerInfo info;
        m_profilerVK.getTimerInfo(nullptr, info);
        ImGui::Text("Frame       [ms]: %2.1f", info.cpu.average / 1000.0f);
        m_profilerVK.getTimerInfo("Upload", info);
        ImGui::Text("Upload      [ms]: %2.3f", info.gpu.average / 1000.0f);

        VkDeviceSize allocSize;
        VkDeviceSize usedSize;
        float        util;
        util = m_memAllocator.getUtilization(allocSize, usedSize);
        ImGui::Text("Total Memory [KB]: %6d", uint32_t(allocSize / 1024));
        ImGui::ProgressBar(util, ImVec2(0.0f, 0.0f));
      }
    }
    ImGui::End();
  }
};

//////////////////////////////////////////////////////////////////////////

// only required due to custom window initialization
// normally we hide this in the various "app*" utility classes

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <vulkan/vulkan_win32.h>
#elif defined LINUX
#include <GLFW/glfw3.h>
#endif

class SampleWindow : public NVPWindow
{
public:
  SampleWindow(Sample& sample)
      : m_sample(sample)
  {
  }

  Sample& m_sample;

  VkInstance m_instance    = VK_NULL_HANDLE;
  VkDevice   m_device      = VK_NULL_HANDLE;
  VkQueue    m_queue       = VK_NULL_HANDLE;
  uint32_t   m_queueFamily = ~0;

  VkSurfaceKHR    m_surface = VK_NULL_HANDLE;
  nvvk::SwapChain m_swapChain;


  bool init(nvvk::Context& context, uint32_t width, uint32_t height)
  {
    m_instance = context.m_instance;
    m_device   = context.m_device;

    // open window
    if(!open(16, 16, width, height, PROJECT_NAME, false))
    {
      return false;
    }
    // setup surface and swapchain (also normally handled by app* class)
    {
      VkResult result;
#ifdef _WIN32
      HWND      hWnd      = glfwGetWin32Window(m_internal);
      HINSTANCE hInstance = GetModuleHandle(NULL);

      VkWin32SurfaceCreateInfoKHR createInfo = {};
      createInfo.sType                       = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
      createInfo.pNext                       = NULL;
      createInfo.hinstance                   = hInstance;
      createInfo.hwnd                        = hWnd;
      result                                 = vkCreateWin32SurfaceKHR(m_instance, &createInfo, nullptr, &m_surface);
#else   // _WIN32
      result = glfwCreateWindowSurface(m_instance, m_internal, NULL, &m_surface);
#endif  // _WIN32
      assert(result == VK_SUCCESS);

      // update our context's default queue to be presentable
      context.setGCTQueueWithPresent(m_surface);

      m_queue       = context.m_queueGCT;
      m_queueFamily = context.m_queueGCT.familyIndex;

      m_swapChain.init(m_device, context.m_physicalDevice, context.m_queueGCT, context.m_queueGCT.familyIndex, m_surface);
      m_swapChain.update(getWidth(), getHeight(), false);

      submitUpdateBarriers();
    }

    return true;
  }

  void submitUpdateBarriers()
  {
    nvvk::ScopeCommandBuffer cmd(m_device, m_queueFamily, m_queue);
    m_swapChain.cmdUpdateBarriers(cmd);
  }

  void deinit()
  {
    m_swapChain.deinit();
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
  }

  // resize swapchain and framebuffer here
  void onWindowResize(int width, int height) override
  {
    if(!m_device || width == 0 || height == 0)
      return;

    vkQueueWaitIdle(m_queue);
    m_swapChain.update(width, height, false);

    // window resize is pretty heavy, let's not care about blocking operations here
    submitUpdateBarriers();

    m_sample.updateFrameBuffer(width, height);
  }

  void onMouseButton(MouseButton button, ButtonAction action, int mods, int x, int y) override
  {
    ImGuiH::mouse_button(button, action);
  }

  void onMouseMotion(int x, int y) override { ImGuiH::mouse_pos(x, y); }

  void onMouseWheel(int delta) override { ImGuiH::mouse_wheel(delta); }

  void onKeyboardChar(unsigned char key, int mods, int x, int y) { ImGuiH::key_char(key); }

  void onKeyboard(KeyCode key, ButtonAction action, int mods, int x, int y) override
  {
    // reload shaders
    if(key == NVPWindow::KEY_R && action == NVPWindow::BUTTON_PRESS)
    {

      m_sample.reloadShaders();
    }

    ImGuiH::key_button(key, action, mods);
  }
};

//////////////////////////////////////////////////////////////////////////

// Main entry point
int main(int argc, const char** argv)
{
  NVPSystem sys(argv[0], PROJECT_NAME);

  // for illustration purposes context and surface creation
  // are done in main, normally you would wrap this in an app* class

  nvvk::Context context;
  {
    // create context
    nvvk::ContextCreateInfo contextInfo;
    contextInfo.apiMajor = 1;
    contextInfo.apiMinor = 1;
    contextInfo.appTitle = PROJECT_NAME;

    // deal with surface extensions (normally you would hide this in an app* class)
    contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME, false);
#ifdef _WIN32
    contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, false);
#else
    contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME, false);
    contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME, false);
#endif
    contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, false);

    // we make use of this extension when measuring time on the transfer queue
    VkPhysicalDeviceHostQueryResetFeaturesEXT hostResetFeatures = nvvk::make<VkPhysicalDeviceHostQueryResetFeaturesEXT>();
    contextInfo.addDeviceExtension(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME, false, &hostResetFeatures);

    // fake optional extension for illustration
    VkPhysicalDeviceMeshShaderFeaturesNV          meshFeatures = nvvk::make<VkPhysicalDeviceMeshShaderFeaturesNV>();
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures = nvvk::make<VkPhysicalDeviceDescriptorIndexingFeaturesEXT>();
    contextInfo.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, true, &meshFeatures);
    contextInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, true, &indexingFeatures);

    if(!context.init(contextInfo))
    {
      return -1;
    }
  }


  // init other internal resources
  Sample       sample;
  SampleWindow window(sample);

  if(!window.init(context, 640, 480))
  {
    return -1;
  }
  if(!sample.init(context, window.getWidth(), window.getHeight()))
  {
    return -1;
  }

  // main event loop
  while(window.pollEvents())
  {
    // don't attempt to render when minimized
    if(!window.isOpen())
    {
      NVPSystem::waitEvents();
      continue;
    }

    if(!window.m_swapChain.acquire())
    {
      exit(-1);
    }

    sample.processFrame(&window.m_swapChain);

    window.m_swapChain.present(context.m_queueGCT);
  }

  window.deinit();
  sample.deinit();
  context.deinit();
}
