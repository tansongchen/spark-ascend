#include <acl/acl.h>
#include <stdint.h>
#include <stdio.h>

extern void launch_validation_sync_probe(void*, uint8_t*);
#define CHECK(call) do { \
    int status = (call); \
    printf("%s = %d\n", #call, status); fflush(stdout); \
    if (status) return 1; \
} while (0)

int main(void) {
    void* output;
    aclrtStream stream;
    CHECK(aclInit(NULL));
    CHECK(aclrtSetDevice(0));  // logical 0; expose only an idle physical NPU != 0
    CHECK(aclrtCreateStream(&stream));
    CHECK(aclrtMalloc(&output, 256, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(output, 256, 0, 256));
    launch_validation_sync_probe(stream, output);
    puts("launched"); fflush(stdout);
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtFree(output));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    return 0;
}
