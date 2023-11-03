#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool init();
bool initApp();
void mainLoop();
void mainLoopVulkan();
void RenderImGui_Vulkan(VkCommandBuffer commandBuffer);

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);