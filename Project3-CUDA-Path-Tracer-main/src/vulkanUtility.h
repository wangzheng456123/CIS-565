#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <ImGui/imgui_impl_vulkan.h>

#include <vulkan/vulkan.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <random>

#define MAX_FRAMES_IN_FLIGHT 2


class vulkanUtility {
	std::vector<char> readFile(const std::string& fileName) {
		std::ifstream file(fileName, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw(std::runtime_error("failed to open file!"));
		}

		size_t fileSize = file.tellg();

		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}



	struct Vertex
	{
		glm::vec4 inPosition;
		glm::vec2 texCoord;

		static VkVertexInputBindingDescription getBindingDescriptions() {
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding = 0;
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			bindingDescription.stride = sizeof(Vertex);

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
			std::array<VkVertexInputAttributeDescription, 2> attributeDescpritions{};
			attributeDescpritions[0].binding = 0;
			attributeDescpritions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
			attributeDescpritions[0].location = 0;
			attributeDescpritions[0].offset = offsetof(Vertex, inPosition);

			attributeDescpritions[1].binding = 0;
			attributeDescpritions[1].format = VK_FORMAT_R32G32_SFLOAT;
			attributeDescpritions[1].location = 1;
			attributeDescpritions[1].offset = offsetof(Vertex, texCoord);

			return attributeDescpritions;
		}
	};

	int windowWidth = 800;
	int windowHeight = 600;

	const std::string MODEL_PATH = "./viking_room.obj";
	const std::string TEXTURE_PATH = "./viking_room.png";

	GLFWwindow* window;
	VkInstance instance;
	VkQueue graphicQueue;
	VkQueue presentQueue;
	VkDebugUtilsMessengerEXT debugMessenger;
	std::vector<VkExtensionProperties> supportExtensions;
	std::vector<VkLayerProperties> supportLayers;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};
	std::vector<const char*> deviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};
	VkDevice device;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descritorSetLayout;
	VkRenderPass renderPass;
	VkPipeline graphicPipeline;
	std::vector<VkFramebuffer> swapChainFrameBuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0;

	bool framebufferResized = false;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	uint32_t mipLevels;
	VkImage textureImage;
	VkImageView textureImageView;
	VkSampler textureSampler;
	VkDeviceMemory textureImageMemory;
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	std::vector<VkDescriptorSet> descritorSets;
	VkDescriptorPool descriptorPool;
	VkDescriptorPool imGuiPool;

	void* texPixels;
	int texWidth, texHeight, texChannels;

	const std::vector<Vertex> vertices = {
	{{-1.0f, -1.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
	{{1.0f, -1.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{1.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
	{{-1.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}}
	};
	const std::vector<uint32_t> indices = { 0, 1, 3, 3, 1, 2 };

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
#ifdef NDEBUG
	const bool enableValidationLayer = false;
#else 
	const bool enableValidationLayer = true;
#endif 

	void createImGuiPool() {
		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};

		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1000;
		pool_info.poolSizeCount = std::size(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;

		vkCreateDescriptorPool(device, &pool_info, nullptr, &imGuiPool);
	}

	void uploadImGuiFonts() {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
		endSingleTimeCommands(commandBuffer);

		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	VkResult CreateDebugUtilsMessengerEXT(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugUtilsMessengerEXT* debugMessenger
	) {
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

		if (func != nullptr) {
			return func(instance, pCreateInfo, pAllocator, debugMessenger);
		}
		else return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	void DestroyDebugUtilsMessengerEXT(
		VkInstance instance,
		VkDebugUtilsMessengerEXT debugMessenger,
		const VkAllocationCallbacks* pAllocator
	) {
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

		if (func != nullptr) {
			func(instance, debugMessenger, pAllocator);
		}
	}

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() {
			return graphicFamily.has_value() && presentFamily.has_value();
		}
	};
	void cleanupSwapChain() {
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto frameBuffer : swapChainFrameBuffers) {
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}
	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createDepthResources();
		createFrameBuffers();
	}
	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData
	) {
		std::cerr << "Validation Layer:" << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(windowWidth, windowHeight, "Vulkan", nullptr, nullptr);
	}
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferInfo.size = size;
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.usage = usage;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create buffer"));
		}

		VkPhysicalDeviceMemoryProperties memoryProperty{};
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperty);

		VkMemoryRequirements memRequirment{};
		vkGetBufferMemoryRequirements(device, buffer, &memRequirment);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.allocationSize = memRequirment.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirment.memoryTypeBits, properties);
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw(std::runtime_error("failed to allocate device memory"));
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);

	}
	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBuffer commandBuffer;
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.commandBufferCount = 1;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

		if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
			throw(std::runtime_error("failed to allocate transfer command buffer"));
		}

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw(std::runtime_error("failed to begin transfer buffer"));
		}

		return commandBuffer;
	}
	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		if (vkQueueSubmit(graphicQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw(std::runtime_error("failed to submit copy command"));
		}

		vkQueueWaitIdle(graphicQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}
	void copyBuffer(VkBuffer dstBuffer, VkBuffer srcBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyInfo{};
		copyInfo.dstOffset = 0;
		copyInfo.size = size;
		copyInfo.srcOffset = 0;

		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyInfo);

		endSingleTimeCommands(commandBuffer);
	}
	void createImage(uint32_t width, uint32_t height, VkImageUsageFlags usage, VkFormat format, VkImageTiling tiling, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, uint32_t mipLevels, VkSampleCountFlagBits sampleCount) {
		VkImageCreateInfo imageInfo{};
		imageInfo.arrayLayers = 1;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.flags = 0;
		imageInfo.format = format;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.mipLevels = mipLevels;
		imageInfo.samples = sampleCount;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.tiling = tiling;
		imageInfo.usage = usage;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create image"));
		}

		VkPhysicalDeviceMemoryProperties memoryProperty{};
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperty);

		VkMemoryRequirements memRequirment{};
		vkGetImageMemoryRequirements(device, image, &memRequirment);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.allocationSize = memRequirment.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirment.memoryTypeBits, properties);
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw(std::runtime_error("failed to allocate device memory"));
		}

		vkBindImageMemory(device, image, imageMemory, 0);

	}
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy copyInfo{};
		copyInfo.bufferImageHeight = 0;
		copyInfo.bufferOffset = 0;
		copyInfo.bufferRowLength = 0;

		copyInfo.imageExtent.depth = 1;
		copyInfo.imageExtent.height = height;
		copyInfo.imageExtent.width = width;

		copyInfo.imageOffset = { 0, 0, 0 };
		copyInfo.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyInfo.imageSubresource.baseArrayLayer = 0;
		copyInfo.imageSubresource.layerCount = 1;
		copyInfo.imageSubresource.mipLevel = 0;

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyInfo);

		endSingleTimeCommands(commandBuffer);
	}
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlagBits features) {
		for (auto format : candidates) {
			VkFormatProperties properties;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
			if (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & features) == features) {
				return format;
			}
		}
		throw(std::runtime_error("failed to find suitable format!"));
	}
	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}
	VkFormat findDepthFormat() {
		return findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}
	void createColorResources() {
		createImage(swapChainExtent.width, swapChainExtent.height, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, swapChainImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory, 1, msaaSamples);
		colorImageView = createImageView(colorImage, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}
	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory, 1, msaaSamples);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
	}
	void generateMipmaps(VkImage image, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels, VkFormat format) {
		VkFormatProperties formatProperties{};
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw(std::runtime_error("format not support linear sampler"));
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int mipWidth = texWidth;
		int mipHeight = texHeight;

		for (int i = 1; i < mipLevels; i++) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.subresourceRange.baseMipLevel = i - 1;

			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			VkImageBlit imageBlit{};
			imageBlit.dstOffsets[0] = { 0, 0, 0 };
			imageBlit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1 , 1 };
			imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlit.dstSubresource.baseArrayLayer = 0;
			imageBlit.dstSubresource.layerCount = 1;
			imageBlit.dstSubresource.mipLevel = i;

			imageBlit.srcOffsets[0] = { 0, 0, 0 };
			imageBlit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlit.srcSubresource.baseArrayLayer = 0;
			imageBlit.srcSubresource.layerCount = 1;
			imageBlit.srcSubresource.mipLevel = i - 1;

			vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlit, VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.subresourceRange.baseMipLevel = mipLevels - 1;

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		endSingleTimeCommands(commandBuffer);

	}
	void createTextureImage() {

		VkDeviceSize imageSize = texHeight * texHeight * 4;
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memset(data, 0, imageSize);

		struct pixelColor {
			uint8_t r, g, b, a;
		};

		for (int i = 0; i < texWidth * texHeight; i++) {
			((pixelColor*)data)[i].r = 255;
		}

		vkUnmapMemory(device, stagingBufferMemory);

		createImage(texWidth, texHeight, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_TILING_OPTIMAL, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, textureImage, textureImageMemory, mipLevels, VK_SAMPLE_COUNT_1_BIT);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
		generateMipmaps(textureImage, texWidth, texHeight, mipLevels, VK_FORMAT_R8G8B8A8_SRGB);
		// transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	void createTextureImageView() {

		textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

	}
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier imageBarrier{};
		imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageBarrier.image = image;
		imageBarrier.newLayout = newLayout;
		imageBarrier.oldLayout = oldLayout;
		imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBarrier.subresourceRange.baseArrayLayer = 0;
		imageBarrier.subresourceRange.baseMipLevel = 0;
		imageBarrier.subresourceRange.layerCount = 1;
		imageBarrier.subresourceRange.levelCount = mipLevels;

		VkPipelineStageFlagBits srcStage;
		VkPipelineStageFlagBits dstStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

			imageBarrier.srcAccessMask = 0;
			imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

			imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}
		else {
			throw(std::runtime_error("unsupported layout transition"));
		}

		vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);

		endSingleTimeCommands(commandBuffer);
	}
	void createTextureSampler() {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.maxLod = static_cast<float>(mipLevels);
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.minLod = 0.0f;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create texture sampler"));
		}
	}
	void createVertexBuffer() {
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		VkDeviceSize size = vertices.size() * sizeof(Vertex);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data);
		memcpy(data, vertices.data(), size);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(vertexBuffer, stagingBuffer, size);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

	}
	void createIndexBuffer() {
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		VkDeviceSize size = indices.size() * sizeof(uint32_t);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data);
		memcpy(data, indices.data(), size);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(indexBuffer, stagingBuffer, size);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memoryProperty{};
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperty);

		for (int i = 0; i < memoryProperty.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && ((properties & memoryProperty.memoryTypes[i].propertyFlags) == properties)) {
				return i;
			}
		}

		throw(std::runtime_error("failed to find suitable memory"));
	}
	void createSyncObjects() {
		VkSemaphoreCreateInfo semaphoreCreateInfo{};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceCreateInfo{};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceCreateInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw(std::runtime_error("failed to create sync objects"));
			}
		}
	}
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw(std::runtime_error("failed to begin command buffer"));
		}

		std::array<VkClearValue, 2> clearColors{};
		clearColors[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearColors[1].depthStencil = { 1.0, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.clearValueCount = clearColors.size();
		renderPassBeginInfo.framebuffer = swapChainFrameBuffers[imageIndex];
		renderPassBeginInfo.pClearValues = clearColors.data();
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = swapChainExtent;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicPipeline);

		VkViewport viewport{};
		viewport.height = (float)swapChainExtent.height;
		viewport.width = (float)swapChainExtent.width;
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissors{};
		scissors.extent = swapChainExtent;
		scissors.offset = { 0, 0 };
		vkCmdSetScissor(commandBuffer, 0, 1, &scissors);

		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };

		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descritorSets[currentFrame], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, indices.size(), 1, 0, 0, 0);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers[currentFrame]);
		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw(std::runtime_error("failed to record command buffer"));
		}

	}
	void createCommandBuffer() {
		VkCommandBufferAllocateInfo allocInfo{};

		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		allocInfo.commandBufferCount = commandBuffers.size();
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create command buffer"));
		}
	}
	void createDescritorSets() {
		descritorSets.resize(MAX_FRAMES_IN_FLIGHT);
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descritorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
		allocInfo.pSetLayouts = layouts.data();
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;

		if (vkAllocateDescriptorSets(device, &allocInfo, descritorSets.data()) != VK_SUCCESS) {
			throw(std::runtime_error("failed to alloc descritor sets"));
		}

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSampler;

			VkWriteDescriptorSet writeInfo{};

			writeInfo.descriptorCount = 1;
			writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			writeInfo.dstArrayElement = 0;
			writeInfo.dstBinding = 0;
			writeInfo.dstSet = descritorSets[i];
			writeInfo.pBufferInfo = nullptr;
			writeInfo.pImageInfo = &imageInfo;
			writeInfo.pTexelBufferView = nullptr;
			writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

			vkUpdateDescriptorSets(device, 1, &writeInfo, 0, nullptr);

		}
	}
	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 1> poolSize{};

		poolSize[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;
		poolSize[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
		poolInfo.poolSizeCount = poolSize.size();
		poolInfo.pPoolSizes = poolSize.data();
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create descritor pool"));
		}

	}
	void createCommandPool() {
		VkCommandPoolCreateInfo createInfo{};

		QueueFamilyIndices indices = findQueueFamilyIndices(physicalDevice);

		createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		createInfo.queueFamilyIndex = indices.graphicFamily.value();
		createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

		if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create command pool"));
		}
	}
	void createFrameBuffers() {
		swapChainFrameBuffers.resize(swapChainImages.size());

		for (int i = 0; i < swapChainImages.size(); i++) {
			std::array<VkImageView, 3> attachments = {
				colorImageView,
				depthImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo createInfo{};
			createInfo.attachmentCount = attachments.size();
			createInfo.height = swapChainExtent.height;
			createInfo.width = swapChainExtent.width;
			createInfo.layers = 1;
			createInfo.pAttachments = attachments.data();
			createInfo.renderPass = renderPass;
			createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;

			if (vkCreateFramebuffer(device, &createInfo, nullptr, &swapChainFrameBuffers[i]) != VK_SUCCESS) {
				throw(std::runtime_error("failed to create framebuffer"));
			}
		}
	}
	void createRenderPass() {
		std::vector<VkAttachmentDescription> attachments{};

		VkAttachmentDescription colorAttachment{};
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.samples = msaaSamples;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments.push_back(colorAttachment);

		VkAttachmentDescription depthAttachment{};
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		depthAttachment.format = findDepthFormat();
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.samples = msaaSamples;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments.push_back(depthAttachment);

		VkAttachmentDescription colorResolveAttachment{};
		colorResolveAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		colorResolveAttachment.format = swapChainImageFormat;
		colorResolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorResolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorResolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments.push_back(colorResolveAttachment);

		VkAttachmentReference resolveRef{};
		resolveRef.attachment = 2;
		resolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorRef{};
		colorRef.attachment = 0;
		colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthRef{};
		depthRef.attachment = 1;
		depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorRef;
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.pDepthStencilAttachment = &depthRef;
		subpass.pResolveAttachments = &resolveRef;

		VkRenderPassCreateInfo createInfo{};
		createInfo.attachmentCount = attachments.size();
		createInfo.pAttachments = attachments.data();
		createInfo.pSubpasses = &subpass;
		createInfo.subpassCount = 1;
		createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

		VkSubpassDependency subpassDependency{};
		subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		subpassDependency.dstSubpass = 0;
		subpassDependency.srcAccessMask = 0;
		subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;

		createInfo.dependencyCount = 1;
		createInfo.pDependencies = &subpassDependency;

		if (vkCreateRenderPass(device, &createInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create render pass"));
		}
	}
	void createDescriptorSetLayout() {

		VkDescriptorSetLayoutBinding samplerBinding{};
		samplerBinding.binding = 0;
		samplerBinding.descriptorCount = 1;
		samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerBinding.pImmutableSamplers = nullptr;
		samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 1> bingdings = { samplerBinding };

		VkDescriptorSetLayoutCreateInfo descritorInfo{};
		descritorInfo.bindingCount = bingdings.size();
		descritorInfo.pBindings = bingdings.data();
		descritorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		if (vkCreateDescriptorSetLayout(device, &descritorInfo, nullptr, &descritorSetLayout) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create descritor set layout"));
		}
	}
	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("vert.spv");
		auto fragShaderCode = readFile("frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescriptions();
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
		inputAssemblyInfo.primitiveRestartEnable = false;
		inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicInfo{};
		dynamicInfo.dynamicStateCount = dynamicStates.size();
		dynamicInfo.pDynamicStates = dynamicStates.data();
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;


		VkPipelineViewportStateCreateInfo viewportStateInfo{};
		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

		VkPipelineRasterizationStateCreateInfo rastInfo{};
		rastInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rastInfo.depthBiasClamp = 0.0;
		rastInfo.depthBiasConstantFactor = 0.0;
		rastInfo.depthBiasEnable = VK_FALSE;
		rastInfo.depthBiasSlopeFactor = 0.0;
		rastInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rastInfo.lineWidth = 1.0f;
		rastInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rastInfo.rasterizerDiscardEnable = VK_FALSE;
		rastInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

		VkPipelineMultisampleStateCreateInfo multisampleInfo{};
		multisampleInfo.alphaToCoverageEnable = VK_FALSE;
		multisampleInfo.alphaToOneEnable = VK_FALSE;
		multisampleInfo.minSampleShading = 1.0;
		multisampleInfo.pSampleMask = nullptr;
		multisampleInfo.rasterizationSamples = msaaSamples;
		multisampleInfo.sampleShadingEnable = VK_FALSE;
		multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

		VkPipelineColorBlendAttachmentState attachmentBlendState{};
		attachmentBlendState.alphaBlendOp = VK_BLEND_OP_ADD;
		attachmentBlendState.blendEnable = VK_FALSE;
		attachmentBlendState.colorBlendOp = VK_BLEND_OP_ADD;
		attachmentBlendState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		attachmentBlendState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		attachmentBlendState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		attachmentBlendState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		attachmentBlendState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;

		VkPipelineColorBlendStateCreateInfo blendState{};
		blendState.attachmentCount = 1;
		blendState.blendConstants[0] = 0.0;
		blendState.blendConstants[1] = 0.0;
		blendState.blendConstants[2] = 0.0;
		blendState.blendConstants[3] = 0.0;
		blendState.logicOp = VK_LOGIC_OP_COPY;
		blendState.logicOpEnable = VK_FALSE;
		blendState.pAttachments = &attachmentBlendState;
		blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.pPushConstantRanges = nullptr;
		layoutInfo.pSetLayouts = &descritorSetLayout;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.setLayoutCount = 1;
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.back = {};
		depthInfo.depthBoundsTestEnable = VK_FALSE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.front = {};
		depthInfo.maxDepthBounds = 1.0f;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.stencilTestEnable = VK_FALSE;
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

		if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create pipeline layout"));
		}

		VkGraphicsPipelineCreateInfo createInfo{};
		createInfo.basePipelineHandle = VK_NULL_HANDLE;
		createInfo.basePipelineIndex = -1;
		createInfo.layout = pipelineLayout;
		createInfo.pColorBlendState = &blendState;
		createInfo.pDepthStencilState = &depthInfo;
		createInfo.pInputAssemblyState = &inputAssemblyInfo;
		createInfo.pMultisampleState = &multisampleInfo;
		createInfo.pRasterizationState = &rastInfo;
		createInfo.pStages = shaderStages;
		createInfo.pVertexInputState = &vertexInputInfo;
		createInfo.pViewportState = &viewportStateInfo;
		createInfo.renderPass = renderPass;
		createInfo.stageCount = 2;
		createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		createInfo.subpass = 0;
		createInfo.pDynamicState = &dynamicInfo;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &graphicPipeline) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create graphic pipeline"));
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}
	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

		VkShaderModule shaderModule;

		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create shader module"));
		}

		return shaderModule;
	}
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlagBits aspectFlags, uint32_t mipLevels) {
		VkImageView imageView;

		VkImageViewCreateInfo createInfo{};

		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;

		createInfo.format = format;
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

		createInfo.subresourceRange.aspectMask = aspectFlags;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = mipLevels;

		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.image = image;


		if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create swap chain image view"));
		}

		return imageView;
	}
	void createImageViews() {


		swapChainImageViews.resize(swapChainImages.size());

		for (int i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}
	void createSwapChain() {
		SwapChainSupportDetails details = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR format = chooseSwapSurfaceFormat(details.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(details.presentModes);
		VkExtent2D extent = chooseSwapExtent(details.capabilities);

		uint32_t imageCount = 0;

		imageCount = details.capabilities.minImageCount + 1;

		if (details.capabilities.maxImageCount > 0 && imageCount > details.capabilities.maxImageCount) {
			imageCount = details.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.imageColorSpace = format.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageFormat = format.format;
		createInfo.surface = surface;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		createInfo.clipped = VK_TRUE;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.oldSwapchain = VK_NULL_HANDLE;
		createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
		createInfo.minImageCount = imageCount;

		QueueFamilyIndices indices = findQueueFamilyIndices(physicalDevice);
		uint32_t queueFamilyIndices[2] = { indices.graphicFamily.value(), indices.presentFamily.value() };

		if (indices.graphicFamily.value() == indices.presentFamily.value()) {
			createInfo.pQueueFamilyIndices = nullptr;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}
		else {
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		}

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw(std::runtime_error("create swap chain failed"));
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = format.format;
		swapChainExtent = extent;
	}
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create window surface"));
		}
	}
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != (uint32_t)std::numeric_limits<uint32_t>::max) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;

			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		}

	}
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> presentModes) {
		for (const auto& presentMode : presentModes) {
			if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				return presentMode;
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> formats) {
		for (auto const& availableFormat : formats) {
			if (availableFormat.format == VK_FORMAT_R8G8B8A8_SRGB && availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
				return availableFormat;
		}
		return formats[0];
	}
	void createLogicalDevice() {
		QueueFamilyIndices indices;
		indices = findQueueFamilyIndices(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueInfos;
		std::set<uint32_t> queueIndices = { indices.graphicFamily.value(), indices.presentFamily.value() };

		for (auto indice : queueIndices) {
			VkDeviceQueueCreateInfo queueInfo{};
			queueInfo.queueCount = 1;
			queueInfo.queueFamilyIndex = indice;
			queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			float priority = 1.0;
			queueInfo.pQueuePriorities = &priority;
			queueInfos.push_back(queueInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo createInfo{};

		createInfo.pQueueCreateInfos = queueInfos.data();
		createInfo.queueCreateInfoCount = queueInfos.size();
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledLayerCount = 0;

		if (enableValidationLayer) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		createInfo.enabledExtensionCount = deviceExtensions.size();
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw(std::runtime_error("failed to create logical device"));
		}

		vkGetDeviceQueue(device, indices.graphicFamily.value(), 0, &graphicQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}
	void pickPhysicalDevice() {
		uint32_t physicalDeviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

		if (physicalDeviceCount == 0) {
			throw(std::runtime_error("no physical device available"));
		}

		std::vector<VkPhysicalDevice> availablePhysicalDevice(physicalDeviceCount);

		vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, availablePhysicalDevice.data());

		for (const auto& device : availablePhysicalDevice) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
			}
			return;
		}
		if (physicalDevice == VK_NULL_HANDLE) {
			throw(std::runtime_error("no suitable physical device\n"));
		}
	}
	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		VkSampleCountFlags sampleCount = properties.limits.sampledImageColorSampleCounts & properties.limits.sampledImageDepthSampleCounts;

		if (sampleCount & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (sampleCount & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (sampleCount & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (sampleCount & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (sampleCount & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (sampleCount & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}
	bool isDeviceSuitable(const VkPhysicalDevice& device) {
		QueueFamilyIndices indices = findQueueFamilyIndices(device);

		bool extensionSupported = checkDeviceExtentiosnSupport(device);
		bool swapChainAdequate = false;

		VkPhysicalDeviceFeatures deviceFeatures{};
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		if (extensionSupported) {
			SwapChainSupportDetails details = querySwapChainSupport(device);
			swapChainAdequate = !details.formats.empty() && !details.presentModes.empty();
		}
		return indices.isComplete() && extensionSupported && swapChainAdequate && deviceFeatures.samplerAnisotropy;
	}
	bool checkDeviceExtentiosnSupport(VkPhysicalDevice device) {
		uint32_t deviceExtensionCount = 0;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, nullptr);
		std::vector<VkExtensionProperties> extensionProperties(deviceExtensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, extensionProperties.data());

		std::set<std::string> propertySet(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& property : extensionProperties) {
			propertySet.erase(property.extensionName);
		}

		return propertySet.empty();
	}
	QueueFamilyIndices findQueueFamilyIndices(VkPhysicalDevice device) {
		QueueFamilyIndices indices{};

		uint32_t queueFamilyCount = 0;

		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);

		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilyProperties.data());

		int i = 0;
		VkBool32 presentSupport = false;
		for (const auto& queueFamily : queueFamilyProperties) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphicFamily = i;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport) indices.presentFamily = i;
			i++;
		}
		return indices;
	}
	std::vector<const char*> getRequiredExtensions() {
		const char** glfwExtensions;
		uint32_t glfwExtensionCount = 0;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayer)
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		return extensions;
	}
	void createInstance() {
		if (enableValidationLayer && !checkValidationLayerSupport()) {
			throw std::runtime_error("Validation Layer Not Support");
		}
		VkApplicationInfo appInfo{};
		appInfo.apiVersion = VK_API_VERSION_1_3;
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pApplicationName = "My Vulkan";
		appInfo.pEngineName = "No Engine";
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();

		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (!enableValidationLayer)
			createInfo.enabledLayerCount = 0;
		else {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();

			debugCreateInfo = populateDebugMessengerCreateInfo();
			createInfo.pNext = &debugCreateInfo;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance");
		}

	}
	VkDebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo() {
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

		return createInfo;
	}
	void setupDebugMessenger() {
		VkDebugUtilsMessengerCreateInfoEXT createInfo = populateDebugMessengerCreateInfo();

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("create debug messenger failed");
		}
	}
	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}
	void cleanup() {
		cleanupSwapChain();
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorPool(device, imGuiPool, nullptr);

		vkDestroyDescriptorSetLayout(device, descritorSetLayout, nullptr);

		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);

		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyPipeline(device, graphicPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		vkDestroyDevice(device, nullptr);
		if (enableValidationLayer) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();

	}
	void getSupportExtensionsAndLayers() {
		uint32_t extensionCount;

		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		supportExtensions.resize(extensionCount);

		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, supportExtensions.data());

		uint32_t layerCount = 0;

		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		supportLayers.resize(layerCount);

		vkEnumerateInstanceLayerProperties(&layerCount, supportLayers.data());

	}

	bool checkValidationLayerSupport() {
		bool res = false;
		for (const auto& layerProperty : validationLayers) {
			for (const auto& layer : supportLayers) {
				if (strcmp(layerProperty, layer.layerName) == 0) {
					res = true;
					break;
				}
			}
			if (!res) return res;
		}
		return res;
	}

public:
	void updateTextureImage(void* pixels) {
		VkDeviceSize imageSize = texHeight * texHeight * 4;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, imageSize);
		vkUnmapMemory(device, stagingBufferMemory);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
		generateMipmaps(textureImage, texWidth, texHeight, mipLevels, VK_FORMAT_R8G8B8A8_SRGB);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	VkCommandBuffer getCommandBuffer() {
		return commandBuffers[currentFrame];
	}
	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw(std::runtime_error("failde to acquire image"));
		}

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
		VkSemaphore signalSemaphore[] = { renderFinishSemaphores[currentFrame] };
		submitInfo.pSignalSemaphores = signalSemaphore;
		VkSemaphore waitSemaphore[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.pWaitSemaphores = waitSemaphore;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;

		if (vkQueueSubmit(graphicQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw(std::runtime_error("failed to submit command queue"));
		}

		VkSwapchainKHR swapChains[] = { swapChain };

		VkPresentInfoKHR presentInfo{};
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pWaitSemaphores = &renderFinishSemaphores[currentFrame];
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.swapchainCount = 1;
		presentInfo.waitSemaphoreCount = 1;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			recreateSwapChain();
			framebufferResized = false;
		}
		else if (result != VK_SUCCESS) throw(std::runtime_error("failed to present image"));

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void initVulkan() {
		getSupportExtensionsAndLayers();
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();
		createFrameBuffers();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexBuffer();
		createIndexBuffer();
		createImGuiPool();
		createDescriptorPool();
		createDescritorSets();
		createCommandBuffer();
		createSyncObjects();
	}

	void setTexture(void* data, int width, int height, int channels) {
		texPixels = data;
		texWidth = width;
		texHeight = height;
		texChannels = channels;
	}

	void setWindowSize(int width, int height) {
		windowWidth = width;
		windowHeight = height;
	}

	void setWindow(GLFWwindow* inWindow) {
		window = inWindow;
	}

	static void check_vk_result(VkResult err)
	{
		if (err == 0)
			return;
		fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
		if (err < 0)
			abort();
	}

	void initImGUI() {
		ImGui_ImplVulkan_InitInfo vulkanInit{};

		vulkanInit.Allocator = nullptr;
		vulkanInit.CheckVkResultFn = check_vk_result;
		vulkanInit.ColorAttachmentFormat = swapChainImageFormat;
		vulkanInit.DescriptorPool = imGuiPool;
		vulkanInit.Device = device;
		vulkanInit.ImageCount = MAX_FRAMES_IN_FLIGHT;
		vulkanInit.Instance = instance;
		vulkanInit.MinImageCount = MAX_FRAMES_IN_FLIGHT;
		vulkanInit.MSAASamples = getMaxUsableSampleCount();
		vulkanInit.PhysicalDevice = physicalDevice;
		vulkanInit.PipelineCache = VK_NULL_HANDLE;
		vulkanInit.Queue = graphicQueue;
		QueueFamilyIndices indices = findQueueFamilyIndices(physicalDevice);
		vulkanInit.QueueFamily = indices.graphicFamily.value();
		vulkanInit.Subpass = 0;
		vulkanInit.UseDynamicRendering = VK_FALSE;

		ImGui_ImplVulkan_Init(&vulkanInit, renderPass);

		uploadImGuiFonts();
	}

	VkRenderPass getRenderPass() {
		return renderPass;
	}

	void setTextureSize(int width, int height) {
		texWidth = width;
		texHeight = height;
	}

	void flushDevice() {
		vkDeviceWaitIdle(device);
	}
};