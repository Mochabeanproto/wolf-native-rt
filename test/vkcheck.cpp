// Standalone probe: does 32-bit Vulkan expose hardware ray tracing on this GPU?
// Dynamically loads vulkan-1.dll (32-bit loader in SysWOW64), no import lib.
// Build x86 and x64; compare. Decides in-process Vulkan vs out-of-process DXR bridge.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <vector>

int main()
{
    printf("=== vkcheck (%zu-bit) ===\n", sizeof(void*) * 8);
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) { printf("vulkan-1.dll not found\n"); return 1; }

    auto vkGIPA = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    auto vkCreateInstance = (PFN_vkCreateInstance)vkGIPA(nullptr, "vkCreateInstance");
    if (!vkCreateInstance) { printf("no vkCreateInstance\n"); return 1; }

    VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &ai;
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &inst);
    if (r != VK_SUCCESS) { printf("vkCreateInstance failed (%d)\n", r); return 1; }

    auto vkEnumPhys = (PFN_vkEnumeratePhysicalDevices)vkGIPA(inst, "vkEnumeratePhysicalDevices");
    auto vkGetProps = (PFN_vkGetPhysicalDeviceProperties)vkGIPA(inst, "vkGetPhysicalDeviceProperties");
    auto vkEnumExt  = (PFN_vkEnumerateDeviceExtensionProperties)vkGIPA(inst, "vkEnumerateDeviceExtensionProperties");

    uint32_t n = 0; vkEnumPhys(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n); vkEnumPhys(inst, &n, devs.data());
    printf("%u physical device(s)\n", n);

    const char* want[] = { "VK_KHR_ray_tracing_pipeline", "VK_KHR_acceleration_structure", "VK_KHR_ray_query" };
    for (auto d : devs) {
        VkPhysicalDeviceProperties p; vkGetProps(d, &p);
        printf("  %s\n", p.deviceName);
        uint32_t ec = 0; vkEnumExt(d, nullptr, &ec, nullptr);
        std::vector<VkExtensionProperties> exts(ec); vkEnumExt(d, nullptr, &ec, exts.data());
        for (const char* w : want) {
            bool found = false;
            for (auto& e : exts) if (strcmp(e.extensionName, w) == 0) { found = true; break; }
            printf("      %-32s %s\n", w, found ? "YES" : "no");
        }
    }
    return 0;
}
